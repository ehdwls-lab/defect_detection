#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

constexpr uint8_t LED_PIN = 6;
constexpr uint16_t NUM_PIXELS = 288;

// 밝기 설정
constexpr uint8_t MIN_BRIGHTNESS = 0;
constexpr uint8_t MAX_SAFE_BRIGHTNESS = 80;
constexpr uint8_t BRIGHTNESS_STEP = 1;

// ============================================================
// 강제 OFF 구간 설정
// 원하는 LED 번호로 수정하면 됨
// ============================================================

// OFF 구간 1
constexpr uint16_t OFF_RANGE1_START = 0;
constexpr uint16_t OFF_RANGE1_END   = 1;

// OFF 구간 2
constexpr uint16_t OFF_RANGE2_START = 0;
constexpr uint16_t OFF_RANGE2_END   = 0;


// NeoPixel 객체
Adafruit_NeoPixel pixels(
    NUM_PIXELS,
    LED_PIN,
    NEO_GRB + NEO_KHZ800
);

// 초기 밝기
uint8_t currentBrightness = 80;

// 현재 RGB 색상 저장
uint8_t currentRed = 0;
uint8_t currentGreen = 0;
uint8_t currentBlue = 0;


/**
 * 특정 LED가 강제 OFF 구간에 포함되는지 확인
 */
bool isForcedOff(uint16_t index)
{
    bool range1 =
        (index >= OFF_RANGE1_START &&
         index <= OFF_RANGE1_END);

    bool range2 =
        (index >= OFF_RANGE2_START &&
         index <= OFF_RANGE2_END);

    return range1 || range2;
}


/**
 * 저장된 색상과 밝기를 LED 전체에 적용
 */
void applyLighting()
{
    pixels.setBrightness(currentBrightness);

    for (uint16_t i = 0; i < NUM_PIXELS; i++)
    {
        // ----------------------------------------------------
        // 1. 지정된 두 구간은 무조건 OFF
        // ----------------------------------------------------
        if (isForcedOff(i))
        {
            pixels.setPixelColor(
                i,
                pixels.Color(0, 0, 0)
            );

            continue;
        }

        // ----------------------------------------------------
        // 2. 나머지 LED는 기존 방식대로
        //    짝수 번호 LED만 켜기
        // ----------------------------------------------------
        if (i % 1 == 0)
        {
            pixels.setPixelColor(
                i,
                pixels.Color(
                    currentRed,
                    currentGreen,
                    currentBlue
                )
            );
        }
        else
        {
            // 홀수 번호 LED OFF
            pixels.setPixelColor(
                i,
                pixels.Color(0, 0, 0)
            );
        }
    }

    pixels.show();
}


/**
 * 전체 LED 색상 변경
 */
void setAllColor(uint8_t red, uint8_t green, uint8_t blue)
{
    currentRed = red;
    currentGreen = green;
    currentBlue = blue;

    applyLighting();
}


/**
 * 현재 설정 출력
 */
void printStatus()
{
    Serial.println();
    Serial.println("===== Current LED Status =====");

    Serial.print("Brightness : ");
    Serial.println(currentBrightness);

    Serial.print("RGB        : ");
    Serial.print(currentRed);
    Serial.print(", ");
    Serial.print(currentGreen);
    Serial.print(", ");
    Serial.println(currentBlue);

    Serial.print("OFF Range 1: ");
    Serial.print(OFF_RANGE1_START);
    Serial.print(" ~ ");
    Serial.println(OFF_RANGE1_END);

    Serial.print("OFF Range 2: ");
    Serial.print(OFF_RANGE2_START);
    Serial.print(" ~ ");
    Serial.println(OFF_RANGE2_END);

    Serial.println("==============================");
    Serial.println();
}


/**
 * 명령어 안내
 */
void printHelp()
{
    Serial.println();
    Serial.println("=== NeoPixel Lighting Controller ===");

    Serial.println("w : White");
    Serial.println("r : Red");
    Serial.println("g : Green");
    Serial.println("b : Blue");

    Serial.println("1 : Bluish-white test mode");
    Serial.println("2 : Neutral-white test mode");

    Serial.println("0 : All LEDs off");

    Serial.println("+ : Increase brightness");
    Serial.println("- : Decrease brightness");

    Serial.println("s : Print current status");
    Serial.println("h : Print this help");

    Serial.println();
}


/**
 * 초기 설정
 */
void setup()
{
    Serial.begin(9600);

    pixels.begin();

    // 초기에는 전부 OFF
    pixels.clear();

    pixels.setBrightness(currentBrightness);
    pixels.show();

    Serial.println();
    Serial.println("NeoPixel controller started.");
    Serial.println("LEDs are initially OFF.");

    printHelp();
}


/**
 * 메인 루프
 */
void loop()
{
    // 시리얼 명령 없으면 종료
    if (Serial.available() <= 0)
    {
        return;
    }

    const char command = Serial.read();

    // Enter 입력 시 들어오는 개행 문자 무시
    if (command == '\n' || command == '\r')
    {
        return;
    }


    switch (command)
    {
        // ----------------------------------------------------
        // White
        // ----------------------------------------------------
        case 'w':
        case 'W':

            setAllColor(255, 255, 255);

            Serial.println("Mode: White");
            printStatus();

            break;


        // ----------------------------------------------------
        // Red
        // ----------------------------------------------------
        case 'r':
        case 'R':

            setAllColor(255, 0, 0);

            Serial.println("Mode: Red");
            printStatus();

            break;


        // ----------------------------------------------------
        // Green
        // ----------------------------------------------------
        case 'g':
        case 'G':

            setAllColor(0, 255, 0);

            Serial.println("Mode: Green");
            printStatus();

            break;


        // ----------------------------------------------------
        // Blue
        // ----------------------------------------------------
        case 'b':
        case 'B':

            setAllColor(0, 0, 255);

            Serial.println("Mode: Blue");
            printStatus();

            break;


        // ----------------------------------------------------
        // Bluish White
        // 금속 표면 촬영 시험용
        // ----------------------------------------------------
        case '1':

            setAllColor(120, 120, 255);

            Serial.println("Mode: Bluish White");
            printStatus();

            break;


        // ----------------------------------------------------
        // Neutral White
        // 플라스틱 / 도장면 촬영 시험용
        // ----------------------------------------------------
        case '2':

            setAllColor(255, 255, 255);

            Serial.println("Mode: Neutral White");
            printStatus();

            break;


        // ----------------------------------------------------
        // 전체 OFF
        // ----------------------------------------------------
        case '0':

            setAllColor(0, 0, 0);

            Serial.println("Mode: All LEDs OFF");
            printStatus();

            break;


        // ----------------------------------------------------
        // 밝기 증가
        // ----------------------------------------------------
        case '+':

            if (currentBrightness <=
                MAX_SAFE_BRIGHTNESS - BRIGHTNESS_STEP)
            {
                currentBrightness += BRIGHTNESS_STEP;
            }
            else
            {
                currentBrightness = MAX_SAFE_BRIGHTNESS;
            }

            applyLighting();

            Serial.println("Brightness increased.");
            printStatus();

            break;


        // ----------------------------------------------------
        // 밝기 감소
        // ----------------------------------------------------
        case '-':

            if (currentBrightness >= BRIGHTNESS_STEP)
            {
                currentBrightness -= BRIGHTNESS_STEP;
            }
            else
            {
                currentBrightness = MIN_BRIGHTNESS;
            }

            applyLighting();

            Serial.println("Brightness decreased.");
            printStatus();

            break;


        // ----------------------------------------------------
        // 현재 상태 출력
        // ----------------------------------------------------
        case 's':
        case 'S':

            printStatus();

            break;


        // ----------------------------------------------------
        // 도움말
        // ----------------------------------------------------
        case 'h':
        case 'H':

            printHelp();

            break;


        // ----------------------------------------------------
        // 알 수 없는 명령
        // ----------------------------------------------------
        default:

            Serial.print("Unknown command: ");
            Serial.println(command);

            break;
    }
}