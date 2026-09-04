// 기존 모터 (CD-D508) 핀 설정
const int PUL_PIN1 = 2;
const int DIR_PIN1 = 3;

// 추가된 모터 (MSD-22er2) 핀 설정 (11번: PUL, 10번: DIR)
const int PUL_PIN2 = 11;
const int DIR_PIN2 = 10;

// 모터 속도 (값이 작을수록 빠릅니다)
const int STEP_DELAY = 600; 

void setup()
{
  Serial.begin(115200);

  // 첫 번째 모터 핀 설정
  pinMode(PUL_PIN1, OUTPUT);
  pinMode(DIR_PIN1, OUTPUT);
  digitalWrite(PUL_PIN1, LOW);
  digitalWrite(DIR_PIN1, LOW);

  // 두 번째 모터 핀 설정
  pinMode(PUL_PIN2, OUTPUT);
  pinMode(DIR_PIN2, OUTPUT);
  digitalWrite(PUL_PIN2, LOW);
  digitalWrite(DIR_PIN2, LOW);

  Serial.println("========================");
  Serial.println("DUAL STEP MOTOR TEST MODE");
  Serial.println("========================");
  Serial.println("Usage: F[steps] or B[steps]");
  Serial.println("Example: F5000 (Forward 5000 steps)");
  Serial.println("Example: B2000 (Backward 2000 steps)");
  Serial.println("========================");
}

// 입력받은 스텝 수만큼 모터를 회전시키는 함수 (두 번째 모터 방향 반전)
void moveSteps(long steps, bool isForward)
{
  // 1. 방향 설정 (두 번째 모터는 방향을 반대로 반전)
  if (isForward) {
    digitalWrite(DIR_PIN1, HIGH);
    digitalWrite(DIR_PIN2, LOW);   // 반대 방향으로 회전하도록 LOW 설정
  } else {
    digitalWrite(DIR_PIN1, LOW);
    digitalWrite(DIR_PIN2, HIGH);  // 반대 방향으로 회전하도록 HIGH 설정
  }
  
  // 방향 신호가 안정될 때까지 짧은 대기
  delay(50); 

  // 2. 지정된 스텝 수만큼 두 모터에 동시에 펄스 생성
  for (long i = 0; i < steps; i++) {
    digitalWrite(PUL_PIN1, HIGH);
    digitalWrite(PUL_PIN2, HIGH);
    
    delayMicroseconds(STEP_DELAY);
    
    digitalWrite(PUL_PIN1, LOW);
    digitalWrite(PUL_PIN2, LOW);
    
    delayMicroseconds(STEP_DELAY);
  }

  Serial.println("=== STOP (Target Reached) ===");
  Serial.println();
}

void loop()
{
  if (Serial.available() > 0)
  {
    char cmd = Serial.read();

    if (cmd == '\n' || cmd == '\r') return;

    if (cmd == 'F' || cmd == 'f')
    {
      long steps = Serial.parseInt(); 
      
      if (steps > 0) {
        Serial.print(">>> DUAL FORWARD ");
        Serial.print(steps);
        Serial.println(" steps");
        moveSteps(steps, true);
      }
    }
    else if (cmd == 'B' || cmd == 'b')
    {
      long steps = Serial.parseInt(); 
      
      if (steps > 0) {
        Serial.print("<<< DUAL BACKWARD ");
        Serial.print(steps);
        Serial.println(" steps");
        moveSteps(steps, false);
      }
    }
  }
}
