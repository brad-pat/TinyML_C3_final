#include <Arduino.h>
#include <MicroTFLite.h>
#define MODEL_DATA_IMPLEMENTATION
//#include "cnn_9f_10hz.h"
#include "hybrid_6f_10hz.h"
#include "imu.h"
#include <stdint.h>
#include <math.h>
#include <atomic>

// Give core1 its own stack (prevents the 4KB/4KB split and stack clobbering)
bool core1_separate_stack = true;

static const char* kLabels[] = { "GRZ", "HAY", "LYN", "MOV", "OTH", "STD" };
static constexpr int kNumLabels = sizeof(kLabels) / sizeof(kLabels[0]);
static std::atomic<int> g_stage{0};
static std::atomic<int> g_input_idx{-1};

// --------- IMU globals ----------
IMU_EN_SENSOR_TYPE enMotionSensorType;
IMU_ST_ANGLES_DATA stAngles;
IMU_ST_SENSOR_DATA stGyroRawData;
IMU_ST_SENSOR_DATA stAccelRawData;
IMU_ST_SENSOR_DATA stMagnRawData;

// --------- Sampling parameters ----------
static constexpr uint32_t SAMPLE_INTERVAL_MS = 100;   // 10 Hz = 100 | 5 Hz = 200
static constexpr int FEATURES = 6;
static constexpr int WINDOW_SECONDS = 3;
static constexpr int SAMPLES_PER_WINDOW = (WINDOW_SECONDS * 1000) / SAMPLE_INTERVAL_MS; // 30
static constexpr int INPUT_FLOATS = SAMPLES_PER_WINDOW * FEATURES; // 270

// --------- Tensor arena ----------
constexpr int kTensorArenaSize = 192 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// --------- Ping-pong buffers ----------
alignas(4) static float window_a[INPUT_FLOATS];
alignas(4) static float window_b[INPUT_FLOATS];

// --------- Shared state (core0 <-> core1) ----------
static std::atomic<int> g_ready{-1};     // -1 = none, 0 = A ready, 1 = B ready
static std::atomic<int> g_busy{0};       // 0 idle, 1 core1 running inference
static std::atomic<int> g_pred{-1};      // last predicted class index
static std::atomic<uint32_t> g_inf_us{0}; // last inference time in us
static std::atomic<uint32_t> g_seq{0};   // increments every window produced
static std::atomic<uint32_t> g_done{0};  // increments every inference done
static std::atomic<uint32_t> g_overrun{0}; // count overruns

static inline void read_imu_9axis(float out9[9]) {
  imuDataGet(&stAngles, &stGyroRawData, &stAccelRawData, &stMagnRawData);

  out9[0] = (float)stAccelRawData.s16X;
  out9[1] = (float)stAccelRawData.s16Y;
  out9[2] = (float)stAccelRawData.s16Z;

  out9[3] = (float)stGyroRawData.s16X;
  out9[4] = (float)stGyroRawData.s16Y;
  out9[5] = (float)stGyroRawData.s16Z;

  out9[6] = (float)stMagnRawData.s16X;
  out9[7] = (float)stMagnRawData.s16Y;
  out9[8] = (float)stMagnRawData.s16Z;
}

static void collect_window_into(float* dst) {
  float sample9[9];
  for (int s = 0; s < SAMPLES_PER_WINDOW; s++) {
    uint32_t start = millis();

    read_imu_9axis(sample9);

    int base = s * FEATURES;
    for (int f = 0; f < FEATURES; f++) dst[base + f] = sample9[f];

    uint32_t elapsed = millis() - start;
    if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
  }
}

// ---------------------------
// Core1: inference loop
// ---------------------------
void setup1() {
  // IMPORTANT: do ModelInit on the same core that runs inference
  // (some libs behave better this way).
  // Keep Serial off core1.
  bool ok = ModelInit(model_hybrid_6f_10hz, tensor_arena, kTensorArenaSize);
  if (!ok) {
    // signal failure by setting pred to -2
    g_pred.store(-2, std::memory_order_release);
    while (true) delay(1000);
  }
}

void loop1() {
  g_stage.store(1, std::memory_order_release);

  int buf = g_ready.load(std::memory_order_acquire);
  if (buf < 0) {
    delay(1);
    return;
  }

  g_stage.store(2, std::memory_order_release);

  if (!g_ready.compare_exchange_strong(buf, -1, std::memory_order_acq_rel)) {
    g_stage.store(3, std::memory_order_release);
    return;
  }

  g_stage.store(4, std::memory_order_release);
  g_busy.store(1, std::memory_order_release);

  float* in = (buf == 0) ? window_a : window_b;

  g_stage.store(5, std::memory_order_release);
  for (int i = 0; i < INPUT_FLOATS; i++) {
    g_input_idx.store(i, std::memory_order_release);

    if (!ModelSetInput(in[i], i, true)) {
      g_stage.store(6, std::memory_order_release);
      g_pred.store(-3, std::memory_order_release);
      g_busy.store(0, std::memory_order_release);
      g_done.fetch_add(1, std::memory_order_acq_rel);
      return;
    }
  }

  g_stage.store(7, std::memory_order_release);
  Serial.println("C1 before invoke");
  uint32_t t0 = micros();
  bool ok = ModelRunInference();
  uint32_t t1 = micros();
  Serial.println("C1 after invoke");

  if (!ok) {
    g_stage.store(8, std::memory_order_release);
    g_pred.store(-4, std::memory_order_release);
    g_busy.store(0, std::memory_order_release);
    g_done.fetch_add(1, std::memory_order_acq_rel);
    return;
  }

  g_stage.store(9, std::memory_order_release);

  int best_i = 0;
  float best_v = ModelGetOutput(0);
  for (int i = 1; i < kNumLabels; i++) {
    float v = ModelGetOutput(i);
    if (v > best_v) { best_v = v; best_i = i; }
  }

  g_stage.store(10, std::memory_order_release);

  g_inf_us.store(t1 - t0, std::memory_order_release);
  g_pred.store(best_i, std::memory_order_release);
  g_busy.store(0, std::memory_order_release);
  g_done.fetch_add(1, std::memory_order_acq_rel);
}

// ---------------------------
// Core0: setup + sampling loop
// ---------------------------
void setup() {
  Serial.begin(115200);
  delay(1500);

  imuInit(&enMotionSensorType);
  if (IMU_EN_SENSOR_TYPE_MPU9250 == enMotionSensorType) {
    Serial.println("Motion sensor is MPU9250");
  } else {
    Serial.println("Motion sensor NULL");
    while (true) delay(1000);
  }

  Serial.println("Core0 started. Core1 will init the model.");
}

void loop() {
  static int write_buf = 0;
  float* target = (write_buf == 0) ? window_a : window_b;

  Serial.print("\nCollecting window ");
  Serial.println(write_buf == 0 ? "A..." : "B...");

  collect_window_into(target);

  // If core1 is still busy and no slot is available, count overrun and drop this window.
  // (Alternative: block/wait for core1, but that defeats the point.)
  if (g_ready.load(std::memory_order_acquire) != -1) {
    g_overrun.fetch_add(1, std::memory_order_acq_rel);
    Serial.println("WARNING: dropped window (core1 not keeping up).");
  } else {
    g_ready.store(write_buf, std::memory_order_release);
    g_seq.fetch_add(1, std::memory_order_acq_rel);
  }

  // If an inference finished since last time, print it
  static uint32_t last_done = 0;
  uint32_t done = g_done.load(std::memory_order_acquire);
  if (done != last_done) {
    last_done = done;

    int pred = g_pred.load(std::memory_order_acquire);
    uint32_t us = g_inf_us.load(std::memory_order_acquire);

    if (pred == -2) Serial.println("ERROR: ModelInit failed on core1.");
    else if (pred == -3) Serial.println("ERROR: ModelSetInput failed on core1.");
    else if (pred == -4) Serial.println("ERROR: ModelRunInference failed on core1.");
    else if (pred >= 0 && pred < kNumLabels) {
      Serial.print("Inference time: ");
      Serial.print(us / 1000.0f, 3);
      Serial.print(" ms | Predicted: ");
      Serial.println(kLabels[pred]);
    }

    uint32_t ov = g_overrun.load(std::memory_order_acquire);
    if (ov) {
      Serial.print("Overruns so far: ");
      Serial.println(ov);
    }
  }

  Serial.print("Debug: busy=");
  Serial.print(g_busy.load(std::memory_order_acquire));
  Serial.print(" ready=");
  Serial.print(g_ready.load(std::memory_order_acquire));
  Serial.print(" stage=");
  Serial.print(g_stage.load(std::memory_order_acquire));
  Serial.print(" input_idx=");
  Serial.println(g_input_idx.load(std::memory_order_acquire));


  write_buf ^= 1;
}