#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <Servo.h>

constexpr uint8_t LED_PIN = 6;
constexpr uint16_t NUM_PIXELS = 288;

constexpr uint8_t MIN_BRIGHTNESS = 0;
constexpr uint8_t MAX_SAFE_BRIGHTNESS = 80;
constexpr uint8_t BRIGHTNESS_STEP = 1;

constexpr uint16_t OFF_RANGE1_START = 0;
constexpr uint16_t OFF_RANGE1_END   = 1;
constexpr uint16_t OFF_RANGE2_START = 0;
constexpr uint16_t OFF_RANGE2_END   = 0;

constexpr uint8_t SERVO_PIN = 9;
constexpr uint8_t SERVO_POS_A = 0;
constexpr uint8_t SERVO_POS_B = 90;
constexpr uint16_t SERVO_HOLD_MS = 600;

Adafruit_NeoPixel pixels(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);
Servo sg90;

uint8_t currentBrightness = 80;
uint8_t currentRed = 0;
uint8_t currentGreen = 0;
uint8_t currentBlue = 0;

uint8_t servoAngle = SERVO_POS_B;
unsigned long servoMoveTime = 0;
bool servoAttached = false;

bool isForcedOff(uint16_t index)
{
bool range1 = (index >= OFF_RANGE1_START && index <= OFF_RANGE1_END);
bool range2 = (index >= OFF_RANGE2_START && index <= OFF_RANGE2_END);
return range1 || range2;
}

void applyLighting()
{
pixels.setBrightness(currentBrightness);

for (uint16_t i = 0; i < NUM_PIXELS; i++)
{
    if (isForcedOff(i))
    {
        pixels.setPixelColor(i, pixels.Color(0, 0, 0));
        continue;
    }

    pixels.setPixelColor(
        i,
        pixels.Color(currentRed, currentGreen, currentBlue)
    );
}

pixels.show();


}

void setAllColor(uint8_t red, uint8_t green, uint8_t blue)
{
currentRed = red;
currentGreen = green;
currentBlue = blue;


applyLighting();


}

void servoWrite(uint8_t angle)
{
servoAngle = angle;


if (!servoAttached)
{
    sg90.attach(SERVO_PIN);
    servoAttached = true;
}

sg90.write(servoAngle);
servoMoveTime = millis();

Serial.print("Servo angle: ");
Serial.println(servoAngle);


}

void servoUpdate()
{
if (servoAttached && millis() - servoMoveTime > SERVO_HOLD_MS)
{
sg90.detach();
servoAttached = false;
}
}

void printStatus()
{
Serial.println();
Serial.println("===== Current Status =====");


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

Serial.print("Servo      : ");
Serial.println(servoAngle);

Serial.println("==========================");
Serial.println();


}

void printHelp()
{
Serial.println();
Serial.println("=== LED + Servo Controller ===");


Serial.println("w : White");
Serial.println("r : Red");
Serial.println("g : Green");
Serial.println("b : Blue");
Serial.println("1 : Bluish-white test mode");
Serial.println("2 : Neutral-white test mode");
Serial.println("0 : All LEDs off");
Serial.println("+ : Increase brightness");
Serial.println("- : Decrease brightness");
Serial.println("[ : Servo 0 deg");
Serial.println("] : Servo 90 deg");
Serial.println("s : Print current status");
Serial.println("h : Print this help");

Serial.println();


}

void setup()
{
Serial.begin(115200);


pixels.begin();
pixels.clear();
pixels.setBrightness(currentBrightness);
pixels.show();

sg90.attach(SERVO_PIN);
sg90.write(servoAngle);
servoAttached = true;
servoMoveTime = millis();

Serial.println();
Serial.println("Controller started.");
Serial.println("LEDs are initially OFF.");

printHelp();


}

void loop()
{
servoUpdate();


if (Serial.available() <= 0)
{
    return;
}

const char command = Serial.read();

if (command == '\n' || command == '\r')
{
    return;
}

switch (command)
{
    case 'w':
    case 'W':
        setAllColor(255, 255, 255);
        Serial.println("Mode: White");
        printStatus();
        break;

    case 'r':
    case 'R':
        setAllColor(255, 0, 0);
        Serial.println("Mode: Red");
        printStatus();
        break;

    case 'g':
    case 'G':
        setAllColor(0, 255, 0);
        Serial.println("Mode: Green");
        printStatus();
        break;

    case 'b':
    case 'B':
        setAllColor(0, 0, 255);
        Serial.println("Mode: Blue");
        printStatus();
        break;

    case '1':
        setAllColor(120, 120, 255);
        Serial.println("Mode: Bluish White");
        printStatus();
        break;

    case '2':
        setAllColor(255, 255, 255);
        Serial.println("Mode: Neutral White");
        printStatus();
        break;

    case '0':
        setAllColor(0, 0, 0);
        Serial.println("Mode: All LEDs OFF");
        printStatus();
        break;

    case '+':
        if (currentBrightness <= MAX_SAFE_BRIGHTNESS - BRIGHTNESS_STEP)
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

    case '[':
        servoWrite(SERVO_POS_A);
        break;

    case ']':
        servoWrite(SERVO_POS_B);
        break;

    case 's':
    case 'S':
        printStatus();
        break;

    case 'h':
    case 'H':
        printHelp();
        break;

    default:
        Serial.print("Unknown command: ");
        Serial.println(command);
        break;
}


}
