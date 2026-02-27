#include <MicroTFLite.h>
#define MODEL_DATA_IMPLEMENTATION
#include "model.h"

#include "imu.h"
#include <stdint.h>
#include <math.h>

static const char* kLabels[] = { "GRZ", "HAY", "LYN", "MOV", "OTH", "STD" };
static constexpr int kNumLabels = sizeof(kLabels) / sizeof(kLabels[0]);

// --------- IMU globals (from your printing sketch) ----------
IMU_EN_SENSOR_TYPE enMotionSensorType;
IMU_ST_ANGLES_DATA stAngles;
IMU_ST_SENSOR_DATA stGyroRawData;
IMU_ST_SENSOR_DATA stAccelRawData;
IMU_ST_SENSOR_DATA stMagnRawData;

// --------- Sampling parameters ----------
static constexpr uint32_t SAMPLE_INTERVAL_MS = 100;   // 10 Hz
static constexpr int FEATURES = 9;                    // ax ay az gx gy gz mx my mz
static constexpr int WINDOW_SECONDS = 3;
static constexpr int SAMPLES_PER_WINDOW = (WINDOW_SECONDS * 1000) / SAMPLE_INTERVAL_MS; // 30
static constexpr int INPUT_FLOATS = SAMPLES_PER_WINDOW * FEATURES; // 270

// --------- Tensor arena (you will likely need to increase this) ----------
constexpr int kTensorArenaSize = 96 * 1024;  // start here; increase if ModelInit fails
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// --------- Window buffer ----------
static float window_buf[INPUT_FLOATS];

// Timing
unsigned long previousMillis = 0;

static inline void read_imu_9axis(float out9[9]) {
  imuDataGet(&stAngles, &stGyroRawData, &stAccelRawData, &stMagnRawData);

  // Raw counts (as you trained)
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

void setup() {
  Serial.begin(115200);
  delay(2000);

  imuInit(&enMotionSensorType);

  if (IMU_EN_SENSOR_TYPE_MPU9250 == enMotionSensorType) {
    Serial.println("Motion sensor is MPU9250");
  } else {
    Serial.println("Motion sensor NULL");
    while (true) { delay(1000); }
  }

  Serial.println("Initializing MicroTFLite model...");
  Serial.println("Initializing MicroTFLite model...");

  Serial.print("Arena size: ");
  Serial.println(kTensorArenaSize);

  Serial.print("Model bytes: ");
  Serial.println((unsigned int)model_len);

  Serial.println("Calling ModelInit...");
  delay(50);

  bool ok = ModelInit(model, tensor_arena, kTensorArenaSize);

  Serial.println("Returned from ModelInit.");
  Serial.print("ModelInit ok? ");
  Serial.println(ok ? "YES" : "NO");

  if (!ok) {
    Serial.println("Model initialization failed! (Try larger arena or smaller model)");
    while (true) delay(1000);
  }

  Serial.println("Model initialization done.");
  if (!ModelInit(model, tensor_arena, kTensorArenaSize)) {
    Serial.println("Model initialization failed! (Increase tensor arena?)");
    while (true) { delay(1000); }
  }

  Serial.println("Model initialization done.");
  ModelPrintMetadata();
  ModelPrintInputTensorDimensions();
  ModelPrintOutputTensorDimensions();

  // Optional: sanity check expected input size
  // If MicroTFLite exposes something like ModelGetInputSize(), use it here.
  Serial.print("Expecting input floats: ");
  Serial.println(INPUT_FLOATS);
}

// Collect one full 3-second window at 10 Hz into window_buf
static bool collect_window_blocking() {
  float sample9[9];

  for (int s = 0; s < SAMPLES_PER_WINDOW; s++) {
    unsigned long start = millis();

    read_imu_9axis(sample9);

    // Layout A: [time][features]
    int base = s * FEATURES;
    for (int f = 0; f < FEATURES; f++) {
      window_buf[base + f] = sample9[f];
    }

    // Sleep to maintain 10 Hz
    unsigned long elapsed = millis() - start;
    if (elapsed < SAMPLE_INTERVAL_MS) {
      delay(SAMPLE_INTERVAL_MS - elapsed);
    }
  }

  return true;
}

void loop() {
  Serial.println("\nCollecting 3-second window...");
  if (!collect_window_blocking()) {
    Serial.println("Window collection failed.");
    return;
  }

  // Push the 270 floats into the model input tensor
  // Assumes single input tensor and float input.
  // The hello_world example uses ModelSetInput(value, index, true).
  for (int i = 0; i < INPUT_FLOATS; i++) {
    if (!ModelSetInput(window_buf[i], i, true)) {
      Serial.print("Failed to set input at index ");
      Serial.println(i);
      return;
    }
  }

  unsigned long t0 = micros();


  if (!ModelRunInference()) {
    Serial.println("RunInference Failed!");
    return;
  }

  unsigned long t1 = micros();
  unsigned long inference_us = t1 - t0;
  Serial.print("Inference time: ");
  Serial.print(inference_us / 1000.0f, 3);
  Serial.println(" ms");

  // ---- Read outputs ----
  // If your model outputs probabilities, typically output tensor length = num_classes
  // Print first N outputs (adjust N to your class count)
  Serial.println("Outputs:");

  int best_i = 0;
  float best_v = ModelGetOutput(0);

  for (int i = 0; i < kNumLabels; i++) {
    float v = ModelGetOutput(i);
    Serial.print("  ");
    Serial.print(kLabels[i]);
    Serial.print(": ");
    Serial.println(v, 6);

    if (v > best_v) {
      best_v = v;
      best_i = i;
    }
  }

  Serial.print("Predicted: ");
  Serial.print(kLabels[best_i]);
  Serial.print(" (");
  Serial.print(best_v, 6);
  Serial.println(")");

  // Run once per window (3 seconds) or add a small delay if you like
  // delay(100);
}