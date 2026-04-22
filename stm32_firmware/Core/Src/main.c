/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  ******************************************************************************
  */
/* USER CODE END Header */

#include "main.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim4;
UART_HandleTypeDef huart2;

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM4_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN PV */

// ===== Servo config =====
#define SERVO_MIN_US     1000
#define SERVO_MAX_US     2000
#define SERVO_CENTER_US  1500
#define SERVO_STEP_MAX   400     // clamp por comando
#define SERVO_CH         TIM_CHANNEL_1

static volatile uint16_t g_servo_us = SERVO_CENTER_US;

// ===== UART line parser (IT RX) =====
static volatile uint8_t rx_byte;
static volatile char    line_buf[64];
static volatile uint8_t line_idx   = 0;
static volatile uint8_t line_ready = 0;

// Handshake state
typedef enum { ST_WAIT_HELLO=0, ST_READY } state_t;
static volatile state_t g_state = ST_WAIT_HELLO;

static char txbuf[64];

/* USER CODE END PV */

/* USER CODE BEGIN 0 */

static inline uint16_t clamp_u16(uint16_t x, uint16_t lo, uint16_t hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static inline int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static void uart_send(const char *s)
{
  HAL_UART_Transmit(&huart2, (uint8_t*)s, (uint16_t)strlen(s), 50);
}

static void servo_write_us(uint16_t us)
{
  us = clamp_u16(us, SERVO_MIN_US, SERVO_MAX_US);
  g_servo_us = us;
  __HAL_TIM_SET_COMPARE(&htim4, SERVO_CH, us);
}

static void servo_step(int32_t delta_us)
{
  delta_us = clamp_i32(delta_us, -SERVO_STEP_MAX, SERVO_STEP_MAX);
  int32_t next = (int32_t)g_servo_us + delta_us;
  if (next < SERVO_MIN_US) next = SERVO_MIN_US;
  if (next > SERVO_MAX_US) next = SERVO_MAX_US;
  servo_write_us((uint16_t)next);
}

// Procesa una línea ya terminada (sin \r\n)
static void process_line(const char *line)
{
  // ---- Handshake ----
  if (g_state == ST_WAIT_HELLO)
  {
    if (strcmp(line, "HELLO") == 0) {
      g_state = ST_READY;
      uart_send("HELLO_OK\n");
    } else {
      uart_send("ERR EXPECT_HELLO\n");
    }
    return;
  }

  // ---- Commands (ready) ----
  // Centro rápido
  if (strcmp(line, "C") == 0) {
    servo_write_us(SERVO_CENTER_US);
    uart_send("OK\n");
    return;
  }

  // d <N>  -> derecha (suma us)
  // i <N>  -> izquierda (resta us)
  // Ej: "d 20" o "i 50"
  if ((line[0] == 'd' || line[0] == 'i') && line[1] == ' ')
  {
    int n = atoi(&line[2]);   // tolera "d 20", "d 200"
    if (n < 0) n = -n;        // por si mandan negativo

    if (line[0] == 'd') servo_step(+n);
    else                servo_step(-n);

    uart_send("OK\n");
    return;
  }

  // P <us>  -> set pulso directo (debug útil)
  if (line[0] == 'P' && line[1] == ' ')
  {
    int us = atoi(&line[2]);
    if (us < SERVO_MIN_US || us > SERVO_MAX_US) {
      uart_send("ERR RANGE\n");
      return;
    }
    servo_write_us((uint16_t)us);
    uart_send("OK\n");
    return;
  }

  uart_send("ERR UNKNOWN\n");
}

/* USER CODE END 0 */

int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_TIM4_Init();
  MX_USART2_UART_Init();

  // ===== Start PWM =====
  HAL_TIM_PWM_Start(&htim4, SERVO_CH);
  servo_write_us(SERVO_CENTER_US);

  // ===== Start UART RX IT =====
  HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);

  uart_send("READY\n");

  while (1)
  {
    if (line_ready)
    {
      line_ready = 0;
      char local[64];
      strncpy(local, (const char*)line_buf, sizeof(local));
      local[sizeof(local)-1] = '\0';
      process_line(local);
    }
  }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 84;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) Error_Handler();
}

static void MX_TIM4_Init(void)
{
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 83;            // 84MHz/84 = 1MHz -> 1 tick = 1us
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 19999;            // 20ms
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim4) != HAL_OK) Error_Handler();

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK) Error_Handler();

  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = SERVO_CENTER_US;    // arranca centrado
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();

  HAL_TIM_MspPostInit(&htim4);
}

static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK) Error_Handler();
}

static void MX_GPIO_Init(void)
{
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
}

// ===== UART RX callback: arma líneas por \n/\r =====
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    uint8_t b = rx_byte;

    if (b == '\n' || b == '\r') {
      if (line_idx > 0) {
        line_buf[line_idx] = '\0';
        line_ready = 1;
        line_idx = 0;
      }
    } else {
      if (line_idx < (sizeof(line_buf) - 1)) line_buf[line_idx++] = (char)b;
      else line_idx = 0;
    }

    HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);
  }
}

void Error_Handler(void)
{
  __disable_irq();
  while (1) { }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) { }
#endif
