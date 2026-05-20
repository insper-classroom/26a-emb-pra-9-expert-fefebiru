#include "FreeRTOS.h"
#include "ei_accelerometer.h"
#include "ei_analogsensor.h"
#include "ei_at_handlers.h"
#include "ei_classifier_porting.h"
#include "ei_device_raspberry_rp2xxx.h"
#include "ei_dht11sensor.h"
#include "ei_inertialsensor.h"
#include "ei_rp2xxx_internal_temperature.h"
#include "ei_run_impulse.h"
#include "ei_ultrasonicsensor.h"
#include "pico/multicore.h"
#include "pico/stdlib.h"
#include "task.h"
#include <stdio.h>
#include <time.h>

#include <hardware/gpio.h>
#include <hardware/i2c.h>
#include <hardware/uart.h>
#include <pico/stdio.h>

#include <FreeRTOS.h>
#include <queue.h>
#include <semphr.h>
#include <stdlib.h>
#include <task.h>

#include "mpu6050.h"
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "model-parameters/model_metadata.h"

using namespace ei;

extern "C" EI_IMPULSE_ERROR
run_classifier(ei::signal_t *signal, ei_impulse_result_t *result, bool debug);

static bool debug_nn = false;

const int MPU_ADDRESS   = 0x68;
const int I2C_SDA_GPIO  = 4;
const int I2C_SCL_GPIO  = 5;

// LED RGB
const int LED_R = 11; // idle   → vermelho
const int LED_G = 12; // updown → verde
const int LED_B = 13; // wave   → azul

static void led_init()
{
    gpio_init(LED_R); gpio_set_dir(LED_R, GPIO_OUT);
    gpio_init(LED_G); gpio_set_dir(LED_G, GPIO_OUT);
    gpio_init(LED_B); gpio_set_dir(LED_B, GPIO_OUT);
}

static void led_set(bool r, bool g, bool b)
{
    gpio_put(LED_R, r);
    gpio_put(LED_G, g);
    gpio_put(LED_B, b);
}

static void mpu6050_init()
{
    i2c_init(i2c_default, 400 * 1000);
    gpio_set_function(I2C_SDA_GPIO, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_GPIO, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_GPIO);
    gpio_pull_up(I2C_SCL_GPIO);

    uint8_t buf[] = { 0x6B, 0x00 };
    i2c_write_blocking(i2c_default, MPU_ADDRESS, buf, 2, false);
}

static void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3], int16_t *temp)
{
    uint8_t buffer[14];
    uint8_t val = 0x3B;
    i2c_write_blocking(i2c_default, MPU_ADDRESS, &val, 1, true);
    i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 14, false);

    for (int i = 0; i < 3; i++)
        accel[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);

    *temp = buffer[6] << 8 | buffer[7];

    for (int i = 0; i < 3; i++)
        gyro[i] = (buffer[8 + i * 2] << 8 | buffer[8 + (i * 2) + 1]);
}

static void gesture_recognize_task(void *p)
{
    led_init();
    mpu6050_init();
    int16_t accelerometer[3], gyro[3], temp;

    while (true) {
        float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

        for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
            mpu6050_read_raw(accelerometer, gyro, &temp);
            buffer[ix + 0] = accelerometer[0];
            buffer[ix + 1] = accelerometer[1];
            buffer[ix + 2] = accelerometer[2];
            vTaskDelay(pdMS_TO_TICKS(10));
        }

        ei::signal_t signal;
        int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
        if (err != 0) {
            ei_printf("Failed to create signal from buffer (%d)\n", err);
            continue;
        }

        ei_impulse_result_t result = { 0 };
        err = run_classifier(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", err);
            continue;
        }

        // Encontra o label com maior confidence
        float max_val = 0;
        const char *max_label = "";
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > max_val) {
                max_val = result.classification[ix].value;
                max_label = result.classification[ix].label;
            }
        }

        // Atualiza LED baseado no gesto detectado
        if (strcmp(max_label, "idle") == 0)
            led_set(1, 0, 0);       // vermelho
        else if (strcmp(max_label, "wave") == 0)
            led_set(0, 0, 1);       // azul
        else if (strcmp(max_label, "updown") == 0)
            led_set(0, 1, 0);       // verde
        else
            led_set(0, 0, 0);       // apagado (snake ou indefinido)

        // Print predições
        ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.):\n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: %.5f\n",
                result.classification[ix].label,
                result.classification[ix].value);
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
    }
}

int main(void)
{
    stdio_init_all();
    sleep_ms(2000);

    xTaskCreate(gesture_recognize_task, "gesture_task", 32768, NULL, 1, NULL);
    vTaskStartScheduler();

    while (true);
}