/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <string.h>
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
extern UART_HandleTypeDef huart1;

uint32_t last_telemetry_tick = 0;

uint8_t rx_buf[64];      // IMU 수신 버퍼
// IMU 원시 각도 (센서 X축, Y축 회전)
float imu_x = 0.0f;
float imu_y = 0.0f;

float offset_x = 0.0f;
float offset_y = 0.0f;
uint8_t is_homed = 0;

float z_term = 0.0f;

// phi = Roll (좌우 기울기), theta = Pitch (앞뒤 기울기)
float cur_phi = 0.0f;
float cur_theta = 0.0f;

float target_phi = 0.0f;
float target_theta = 0.0f;
float current_z_cm = 0.0f;
float target_z_cm = 0.0f;

float err_phi = 0.0f;
float err_theta = 0.0f;
float prev_err_phi = 0.0f;
float prev_err_theta = 0.0f;

// sleep 모드 변수
uint8_t sleep_mode = 0;    // 0은 동작 중, 1은 대기 중
uint32_t stable_since = 0; // 오차 안정 구간 시작 시각

// 수신 parser 및 터미널 통신 변수
uint8_t rx_data;           // 1바이트 수신 버퍼
char cmd_buf[32];          // 조립된 문장 버퍼
uint8_t cmd_idx = 0;
uint8_t cmd_received = 0;  // 수신 완료 플래그

// Modbus RTU 요청 패킷
// 확장 : 0x37부터 9워드 (자이로 3, 지자기 3, 각도 3)
uint8_t imu_req_ext[8]   = {0x50, 0x03, 0x00, 0x37, 0x00, 0x09, 0x39, 0x83};
// 축소 : 0x3D부터 3워드 (각도 3)
uint8_t imu_req_angle[8] = {0x50, 0x03, 0x00, 0x3D, 0x00, 0x03, 0x99, 0x86};

// IMU 읽기 모드 0 판별중 / 1 자이로 포함 / 2 각도만
uint8_t imu_mode = 0;
uint8_t imu_ext_fail = 0;

// 자이로 각속도 deg/s
float gyro_x = 0.0f;
float gyro_y = 0.0f;
float rate_phi = 0.0f;
float rate_theta = 0.0f;
float gyro_off_x = 0.0f;
float gyro_off_y = 0.0f;

#define GYRO_LPF_A   0.15f
#define GYRO_SIGN    (+1.0f)   // 각속도 부호가 각도와 반대면 -1.0f

// 제어 게인 : 단위는 deg 당 PWM, (deg/s) 당 PWM
#define KP_THETA      90.0f
#define KP_PHI        105.0f
#define KDR_THETA     2.90f
#define KDR_PHI       3.30f

// 자이로 실패 시 차분 미분 게인
#define KD_THETA      15.0f
#define KD_PHI        12.0f

// 제어 문턱값
#define ERR_DEADBAND   0.10f    // PD 동작 하한
#define SLEEP_ENTER    0.15f    // 슬립 진입 오차
#define SLEEP_EXIT     0.25f    // 슬립 해제 오차
#define SLEEP_HOLD_MS  300      // 슬립 진입 유지 시간
#define OUT_MIN        0.5f     // 출력 무시 하한
#define PWM_MIN        220
#define PWM_MAX        700

// 미세 구동 펄스
#define PULSE_FULL     50.0f    // 이 이상이면 연속 구동
#define PULSE_ON_TICKS 2        // 한 펄스의 최소 구동 틱
#define PULSE_DUTY_MIN 0.20f    // 최저 듀티

// 루프 및 통신 주기
#define LOOP_DELAY_MS  8
#define TLM_PERIOD_MS  20

// 좌표계 : X축 = 전방, Y축 = 좌측, Z축 = 상방
// P1 = (Rp, 0), P2 = (-0.5Rp, +0.866Rp), P3 = (-0.5Rp, -0.866Rp)
// Pgiz = dz - theta * Pix + phi * Piy
#define K_THETA_P1    1.0000f
#define K_THETA_P23   0.5000f
#define K_PHI_P23     0.8660f

// 각도 부호 정의 및 IMU 장착 방향 보정
// phi   (+) : 3번(우) 상승, 2번(좌) 하강
// theta (+) : 1번(앞) 상승, 2번 3번 하강
#define SIGN_PHI      (+1.0f)
#define SIGN_THETA    (+1.0f)

// 원점 재설정 진행 상태
uint8_t homing_active = 0;
uint8_t has_offset = 0;

int16_t mo1 = 0, mo2 = 0, mo3 = 0;

// 제어기 모드 0 = P, 1 = PD, 2 = PID
uint8_t ctrl_mode = 1;

float integ_phi = 0.0f;
float integ_theta = 0.0f;
uint32_t last_ctrl_tick = 0;

#define KI_PHI        31.0f
#define KI_THETA      27.0f
#define INTEG_LIMIT   200.0f

float pulse_acc[3] = {0.0f, 0.0f, 0.0f};
uint8_t pulse_on[3] = {0, 0, 0};
int8_t pulse_dir[3] = {0, 0, 0};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_TIM4_Init(void);
/* USER CODE BEGIN PFP */
void Set_Motor(uint8_t motor_num, uint8_t dir, uint16_t speed);
void Run_Homing(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void Integ_Reset(void)
{
    integ_phi = 0.0f;
    integ_theta = 0.0f;
    last_ctrl_tick = 0;
}

static void Pulse_Reset(void)
{
    for (int i = 0; i < 3; i++) {
        pulse_acc[i] = 0.0f;
        pulse_on[i] = 0;
        pulse_dir[i] = 0;
    }
}

static int16_t Motor_Command(int idx, float out)
{
    if (pulse_on[idx] > 0) {
        pulse_on[idx]--;
        return (pulse_dir[idx] > 0) ? (int16_t)PWM_MIN : (int16_t)(-PWM_MIN);
    }

    float a = fabsf(out);

    if (a <= OUT_MIN) {
        pulse_acc[idx] = 0.0f;
        return 0;
    }

    int8_t dir = (out > 0.0f) ? 1 : -1;

    if (a >= PULSE_FULL) {
        pulse_acc[idx] = 0.0f;

        float p = (float)PWM_MIN + (a - PULSE_FULL);
        if (p > (float)PWM_MAX) p = (float)PWM_MAX;

        return (dir > 0) ? (int16_t)p : (int16_t)(-p);
    }

    float duty = a / PULSE_FULL;
    if (duty < PULSE_DUTY_MIN) duty = PULSE_DUTY_MIN;

    pulse_acc[idx] += duty;

    if (pulse_acc[idx] >= (float)PULSE_ON_TICKS) {
        pulse_acc[idx] -= (float)PULSE_ON_TICKS;
        pulse_dir[idx] = dir;
        pulse_on[idx] = PULSE_ON_TICKS - 1;
        return (dir > 0) ? (int16_t)PWM_MIN : (int16_t)(-PWM_MIN);
    }

    return 0;
}

static void Apply_Motor(uint8_t num, int16_t val)
{
    if (val > 0)      Set_Motor(num, 1, (uint16_t)val);
    else if (val < 0) Set_Motor(num, 0, (uint16_t)(-val));
    else              Set_Motor(num, 1, 0);
}

static void Read_IMU(void)
{
    uint8_t got_gyro = 0;

    if (imu_mode != 2) {
        __HAL_UART_CLEAR_OREFLAG(&huart1);
        huart1.RxState = HAL_UART_STATE_READY;

        HAL_UART_Transmit(&huart1, imu_req_ext, 8, 20);
        if (HAL_UART_Receive(&huart1, rx_buf, 23, 25) == HAL_OK
            && rx_buf[0] == 0x50 && rx_buf[1] == 0x03 && rx_buf[2] == 18) {

            int16_t raw_gx = (int16_t)((rx_buf[3]  << 8) | rx_buf[4]);
            int16_t raw_gy = (int16_t)((rx_buf[5]  << 8) | rx_buf[6]);
            int16_t raw_ax = (int16_t)((rx_buf[15] << 8) | rx_buf[16]);
            int16_t raw_ay = (int16_t)((rx_buf[17] << 8) | rx_buf[18]);

            gyro_x = (float)raw_gx / 32768.0f * 2000.0f;
            gyro_y = (float)raw_gy / 32768.0f * 2000.0f;
            imu_x  = (float)raw_ax / 32768.0f * 180.0f;
            imu_y  = (float)raw_ay / 32768.0f * 180.0f;

            imu_mode = 1;
            imu_ext_fail = 0;
            got_gyro = 1;
        } else {
            if (imu_ext_fail < 255) imu_ext_fail++;
            if (imu_mode == 0 && imu_ext_fail >= 10) imu_mode = 2;
        }
    }

    if (!got_gyro) {
        __HAL_UART_CLEAR_OREFLAG(&huart1);
        huart1.RxState = HAL_UART_STATE_READY;

        HAL_UART_Transmit(&huart1, imu_req_angle, 8, 20);
        if (HAL_UART_Receive(&huart1, rx_buf, 11, 20) == HAL_OK) {
            int16_t raw_x = (int16_t)((rx_buf[3] << 8) | rx_buf[4]);
            int16_t raw_y = (int16_t)((rx_buf[5] << 8) | rx_buf[6]);
            imu_x = (float)raw_x / 32768.0f * 180.0f;
            imu_y = (float)raw_y / 32768.0f * 180.0f;
        }
    }

    cur_phi   = SIGN_PHI   * (imu_x - offset_x);
    cur_theta = SIGN_THETA * (imu_y - offset_y);

    err_phi   = target_phi   - cur_phi;
    err_theta = target_theta - cur_theta;

    if (got_gyro) {
        float r_phi   = GYRO_SIGN * SIGN_PHI   * (gyro_x - gyro_off_x);
        float r_theta = GYRO_SIGN * SIGN_THETA * (gyro_y - gyro_off_y);

        rate_phi   += GYRO_LPF_A * (r_phi   - rate_phi);
        rate_theta += GYRO_LPF_A * (r_theta - rate_theta);
    }
}

static void Angle_PD_Output(void)
{
    float fix_phi = 0.0f;
    float fix_theta = 0.0f;

    uint32_t now = HAL_GetTick();
    float dt = (last_ctrl_tick == 0) ? 0.020f : (float)(now - last_ctrl_tick) / 1000.0f;
    last_ctrl_tick = now;
    if (dt > 0.100f) dt = 0.100f;

    float damp_phi;
    float damp_theta;

    if (imu_mode == 1) {
        damp_phi   = -(rate_phi   * KDR_PHI);
        damp_theta = -(rate_theta * KDR_THETA);

        prev_err_phi = err_phi;
        prev_err_theta = err_theta;
    } else {
        damp_phi   = (err_phi   - prev_err_phi)   * KD_PHI;
        damp_theta = (err_theta - prev_err_theta) * KD_THETA;

        prev_err_phi = err_phi;
        prev_err_theta = err_theta;
    }

    if (fabsf(err_phi) > ERR_DEADBAND) {
        fix_phi = err_phi * KP_PHI;
    }

    if (fabsf(err_theta) > ERR_DEADBAND) {
        fix_theta = err_theta * KP_THETA;
    }

    if (ctrl_mode != 0) {
        fix_phi   += damp_phi;
        fix_theta += damp_theta;
    }

    if (ctrl_mode == 2) {
        float lim_phi   = INTEG_LIMIT / KI_PHI;
        float lim_theta = INTEG_LIMIT / KI_THETA;

        integ_phi   += err_phi   * dt;
        integ_theta += err_theta * dt;

        if (integ_phi >  lim_phi)   integ_phi =  lim_phi;
        if (integ_phi < -lim_phi)   integ_phi = -lim_phi;
        if (integ_theta >  lim_theta) integ_theta =  lim_theta;
        if (integ_theta < -lim_theta) integ_theta = -lim_theta;

        fix_phi   += integ_phi   * KI_PHI;
        fix_theta += integ_theta * KI_THETA;
    } else {
        integ_phi = 0.0f;
        integ_theta = 0.0f;
    }

    float out1 =  (fix_theta * K_THETA_P1);
    float out2 = -(fix_theta * K_THETA_P23) - (fix_phi * K_PHI_P23);
    float out3 = -(fix_theta * K_THETA_P23) + (fix_phi * K_PHI_P23);

    mo1 = Motor_Command(0, out1);
    mo2 = Motor_Command(1, out2);
    mo3 = Motor_Command(2, out3);
}

static void Send_Telemetry(void)
{
    char tx_msg[144];
    int len = snprintf(tx_msg, sizeof(tx_msg),
                       "TLM:Z=%.2f,R=%.2f,P=%.2f,S=%d,M1=%d,M2=%d,M3=%d,H=%d,G=%d,C=%d,VR=%.2f,VP=%.2f\r\n",
                       current_z_cm, cur_phi, cur_theta, sleep_mode,
                       mo1, mo2, mo3, homing_active, imu_mode, ctrl_mode,
                       rate_phi, rate_theta);

    HAL_UART_Transmit(&huart2, (uint8_t*)tx_msg, len, 20);
}

static void Homing_Delay(uint32_t ms)
{
    uint32_t start = HAL_GetTick();

    while ((HAL_GetTick() - start) < ms) {
        HAL_Delay(20);

        if (HAL_GetTick() - last_telemetry_tick >= TLM_PERIOD_MS) {
            last_telemetry_tick = HAL_GetTick();
            Send_Telemetry();
        }
    }
}

static void Level_Platform(uint32_t timeout_ms)
{
    if (!has_offset) return;

    target_phi = 0.0f;
    target_theta = 0.0f;
    prev_err_phi = 0.0f;
    prev_err_theta = 0.0f;
    Pulse_Reset();

    uint32_t start = HAL_GetTick();
    uint32_t ok_since = 0;

    while ((HAL_GetTick() - start) < timeout_ms) {
        Read_IMU();
        Angle_PD_Output();

        Apply_Motor(1, mo1);
        Apply_Motor(2, mo2);
        Apply_Motor(3, mo3);

        if (fabsf(err_phi) <= SLEEP_ENTER && fabsf(err_theta) <= SLEEP_ENTER) {
            if (ok_since == 0) ok_since = HAL_GetTick();
            if ((HAL_GetTick() - ok_since) >= SLEEP_HOLD_MS) break;
        } else {
            ok_since = 0;
        }

        if (HAL_GetTick() - last_telemetry_tick >= TLM_PERIOD_MS) {
            last_telemetry_tick = HAL_GetTick();
            Send_Telemetry();
        }

        HAL_Delay(LOOP_DELAY_MS);
    }

    mo1 = 0; mo2 = 0; mo3 = 0;
    Apply_Motor(1, 0);
    Apply_Motor(2, 0);
    Apply_Motor(3, 0);
    Homing_Delay(500);
}

void Run_Homing(void)
{
    homing_active = 1;
    is_homed = 0;
    sleep_mode = 0;
    stable_since = 0;
    Pulse_Reset();
    Integ_Reset();

    Level_Platform(8000);

    mo1 = -1000; mo2 = -1000; mo3 = -1000;
    Apply_Motor(1, mo1);
    Apply_Motor(2, mo2);
    Apply_Motor(3, mo3);
    Homing_Delay(10000);

    mo1 = 1000; mo2 = 1000; mo3 = 1000;
    Apply_Motor(1, mo1);
    Apply_Motor(2, mo2);
    Apply_Motor(3, mo3);
    Homing_Delay(500);

    mo1 = 0; mo2 = 0; mo3 = 0;
    Apply_Motor(1, mo1);
    Apply_Motor(2, mo2);
    Apply_Motor(3, mo3);
    Homing_Delay(1000);

    __HAL_UART_CLEAR_OREFLAG(&huart1);
    __HAL_UART_CLEAR_NEFLAG(&huart1);
    __HAL_UART_CLEAR_FEFLAG(&huart1);
    huart1.ErrorCode = HAL_UART_ERROR_NONE;
    huart1.gState = HAL_UART_STATE_READY;
    huart1.RxState = HAL_UART_STATE_READY;

    while (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE)) {
        uint8_t dummy = (uint8_t)(huart1.Instance->DR & 0x00FF);
        (void)dummy;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_gx = 0.0f;
    float sum_gy = 0.0f;
    int sample_count = 0;

    gyro_off_x = 0.0f;
    gyro_off_y = 0.0f;

    for (int i = 0; i < 50; i++) {
        float prev_ix = imu_x;
        float prev_iy = imu_y;

        Read_IMU();

        if (imu_x != prev_ix || imu_y != prev_iy || i == 0) {
            sum_x  += imu_x;
            sum_y  += imu_y;
            sum_gx += gyro_x;
            sum_gy += gyro_y;
            sample_count++;
        }

        HAL_Delay(20);

        if (HAL_GetTick() - last_telemetry_tick >= TLM_PERIOD_MS) {
            last_telemetry_tick = HAL_GetTick();
            Send_Telemetry();
        }
    }

    if (sample_count > 0) {
        offset_x = sum_x / (float)sample_count;
        offset_y = sum_y / (float)sample_count;

        if (imu_mode == 1) {
            gyro_off_x = sum_gx / (float)sample_count;
            gyro_off_y = sum_gy / (float)sample_count;
        }

        has_offset = 1;
    }

    imu_x = offset_x;
    imu_y = offset_y;
    rate_phi = 0.0f;
    rate_theta = 0.0f;
    cur_phi = 0.0f;
    cur_theta = 0.0f;
    err_phi = 0.0f;
    err_theta = 0.0f;
    prev_err_phi = 0.0f;
    prev_err_theta = 0.0f;

    current_z_cm = 0.0f;
    target_z_cm = 0.0f;
    target_phi = 0.0f;
    target_theta = 0.0f;

    cmd_idx = 0;
    cmd_received = 0;
    memset(cmd_buf, 0, sizeof(cmd_buf));

    homing_active = 0;
    is_homed = 1;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_USART2_UART_Init();
  MX_USART1_UART_Init();
  MX_TIM4_Init();

  /* USER CODE BEGIN 2 */
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);

  HAL_Delay(500);

  __HAL_UART_CLEAR_OREFLAG(&huart2);
  HAL_UART_Receive_IT(&huart2, &rx_data, 1);

  Run_Homing();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
      if (!is_homed) continue;

      // 1. PC GUI 명령 파서 (Z, R, P 수신 처리)
      if (cmd_received == 1) {
          if (strstr(cmd_buf, "RST") != NULL) {
              memset(cmd_buf, 0, sizeof(cmd_buf));
              cmd_received = 0;
              Run_Homing();
              continue;
          }

          char *ptr_c = strstr(cmd_buf, "MODE:");
          if (ptr_c != NULL) {
              int m = 0;
              if (sscanf(ptr_c, "MODE:%d", &m) == 1 && m >= 0 && m <= 2) {
                  ctrl_mode = (uint8_t)m;
                  Integ_Reset();
                  Pulse_Reset();
              }
          }

          char *ptr_z = strstr(cmd_buf, "Z:");
          if (ptr_z != NULL) sscanf(ptr_z, "Z:%f", &target_z_cm);

          char *ptr_r = strstr(cmd_buf, "R:");
          if (ptr_r != NULL) sscanf(ptr_r, "R:%f", &target_phi);

          char *ptr_p = strstr(cmd_buf, "P:");
          if (ptr_p != NULL) sscanf(ptr_p, "P:%f", &target_theta);

          memset(cmd_buf, 0, sizeof(cmd_buf));
          cmd_received = 0;

          sleep_mode = 0;
          stable_since = 0;
          Integ_Reset();
      }

      // 2. Z축 이동 제어 (상승/하강 속도 분리 적용)
		float speed_up   = 1.300f; // 상승 전용 속도 (cm/s)
		float speed_down = 1.254f; // 하강 전용 속도 (cm/s)

		z_term = 0.0f;
		float z_error = target_z_cm - current_z_cm;
		float active_speed = (z_error > 0) ? speed_up : speed_down;
		float move_per_tick = active_speed * 0.020f;

		if (fabsf(z_error) > move_per_tick) {
			sleep_mode = 0;
			stable_since = 0;

			if (z_error > 0) {
				current_z_cm += move_per_tick;
				z_term = 1000.0f;  // 상승
			} else {
				current_z_cm -= move_per_tick;
				z_term = -1000.0f; // 하강
			}
		} else {
			current_z_cm = target_z_cm;
			z_term = 0.0f;
		}

      // 3. IMU 센서 데이터 수신
      Read_IMU();

      // 4. 수면 및 기상 판단 (Z축 정지 상태일 때만 판단)
      if (z_term == 0.0f) {
          if (sleep_mode == 1) {
              if (fabsf(err_phi) > SLEEP_EXIT || fabsf(err_theta) > SLEEP_EXIT) {
                  sleep_mode = 0;
                  stable_since = 0;
              }
          } else {
              if (fabsf(err_phi) <= SLEEP_ENTER && fabsf(err_theta) <= SLEEP_ENTER) {
                  if (stable_since == 0) stable_since = HAL_GetTick();
                  if ((HAL_GetTick() - stable_since) >= SLEEP_HOLD_MS) {
                      sleep_mode = 1;
                  }
              } else {
                  stable_since = 0;
              }
          }
      }

      // 5. 모터 최종 출력
      mo1 = 0; mo2 = 0; mo3 = 0;

      if (sleep_mode == 0) {
          if (z_term != 0.0f) {
              int16_t v = (z_term > 0) ? 1000 : -1000;
              mo1 = v; mo2 = v; mo3 = v;
              Pulse_Reset();
          } else {
              Angle_PD_Output();
          }
      } else {
          Pulse_Reset();
          Integ_Reset();
      }

      Apply_Motor(1, mo1);
      Apply_Motor(2, mo2);
      Apply_Motor(3, mo3);

      // 6. 실시간 각도 전송 (100ms 주기)
      if (HAL_GetTick() - last_telemetry_tick >= TLM_PERIOD_MS) {
          last_telemetry_tick = HAL_GetTick();
          Send_Telemetry();
      }

      HAL_Delay(LOOP_DELAY_MS);
  }
  /* USER CODE END 3 */
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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_TIM2_Init(void)
{
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 83;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 999;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK) Error_Handler();

  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK) Error_Handler();

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK) Error_Handler();

  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK) Error_Handler();

  HAL_TIM_MspPostInit(&htim2);
}

static void MX_TIM3_Init(void)
{
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 83;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK) Error_Handler();

  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK) Error_Handler();

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK) Error_Handler();

  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();

  HAL_TIM_MspPostInit(&htim3);
}

static void MX_TIM4_Init(void)
{
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_IC_InitTypeDef sConfigIC = {0};

  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 83;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 65535;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_IC_Init(&htim4) != HAL_OK) Error_Handler();

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK) Error_Handler();

  sConfigIC.ICPolarity = TIM_INPUTCHANNELPOLARITY_RISING;
  sConfigIC.ICSelection = TIM_ICSELECTION_DIRECTTI;
  sConfigIC.ICPrescaler = TIM_ICPSC_DIV1;
  sConfigIC.ICFilter = 0;
  if (HAL_TIM_IC_ConfigChannel(&htim4, &sConfigIC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();
}

static void MX_USART1_UART_Init(void)
{
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK) Error_Handler();
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
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOA, LD2_Pin|TRIG_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, DIR1_Pin|DIR2_Pin|DIR3_Pin|GPIO_PIN_5, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LD2_Pin|TRIG_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = DIR1_Pin|DIR2_Pin|DIR3_Pin|GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
}

/* USER CODE BEGIN 4 */
void Set_Motor(uint8_t motor_num, uint8_t dir, uint16_t speed) {
    if (speed > 1000) speed = 1000;

    uint8_t real_dir = dir ? 0 : 1;

    if (motor_num == 1) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_2, real_dir ? GPIO_PIN_SET : GPIO_PIN_RESET);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, speed);
    }
    else if (motor_num == 2) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, real_dir ? GPIO_PIN_SET : GPIO_PIN_RESET);
        __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, speed);
    }
    else if (motor_num == 3) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, real_dir ? GPIO_PIN_SET : GPIO_PIN_RESET);
        __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, speed);
    }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART2) {
        if (rx_data == '\n' || rx_data == '\r') {
            if (cmd_idx > 0) {
                cmd_buf[cmd_idx] = '\0';
                cmd_received = 1;
                cmd_idx = 0;
            }
        }
        else if (rx_data == '\b' || rx_data == 127) {
            if (cmd_idx > 0) cmd_idx--;
        }
        else {
            if (cmd_idx < 31) {
                cmd_buf[cmd_idx++] = rx_data;
            }
        }

        __HAL_UART_CLEAR_OREFLAG(&huart2);
        __HAL_UART_CLEAR_NEFLAG(&huart2);
        __HAL_UART_CLEAR_FEFLAG(&huart2);
        huart2.RxState = HAL_UART_STATE_READY;
        HAL_UART_Receive_IT(&huart2, &rx_data, 1);
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART2) {
        __HAL_UART_CLEAR_OREFLAG(&huart2);
        __HAL_UART_CLEAR_NEFLAG(&huart2);
        __HAL_UART_CLEAR_FEFLAG(&huart2);
        huart2.ErrorCode = HAL_UART_ERROR_NONE;
        huart2.RxState = HAL_UART_STATE_READY;
        HAL_UART_Receive_IT(&huart2, &rx_data, 1);
    }
}
/* USER CODE END 4 */

void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  __disable_irq();
  while (1) {}
  /* USER CODE END Error_Handler_Debug */
}
