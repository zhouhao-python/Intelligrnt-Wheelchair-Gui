#include <Servo.h>
Servo qianhou,zuoyou;
const int pinNumber = 12;
int potpin = 0;
int val;
int qianhou_offset = 0;
int zuoyou_offset = 0;

void setup() {
  // put your setup code here, to run once:
  qianhou.attach(10);
  zuoyou.attach(9);
  Serial.begin(115200); //设置波特率
  pinMode(pinNumber,OUTPUT);
  qianhou.write(90);
  zuoyou.write(70);
}

void loop() {
  if (Serial.available() > 0){

    // put your main code here, to run repeatedly:
    char read_byte = Serial.read();
//    Serial.println(read_byte);
    if (read_byte == 'w'){
      digitalWrite(pinNumber,HIGH);
      qianhou.write(90);
      zuoyou.write(70);
      delay(10);
      qianhou.write(118+qianhou_offset);
      zuoyou.write(87+zuoyou_offset);
      
    }
    if (read_byte == 's'){
      digitalWrite(pinNumber,LOW);
//      qianhou.write(110+qianhou_offset);
//      zuoyou.write(70);
      qianhou.write(70);
      zuoyou.write(70);
    }
    if (read_byte == 'a'){
      digitalWrite(pinNumber,HIGH);
      qianhou.write(90);
      zuoyou.write(70);
      delay(10);
      zuoyou.write(55+zuoyou_offset);
      
    }
    if (read_byte == 'd'){
      digitalWrite(pinNumber,LOW);
      qianhou.write(90);
      zuoyou.write(70);
      delay(10);
      zuoyou.write(92+zuoyou_offset);
    }
    if (read_byte == 'r'){
      digitalWrite(pinNumber,LOW);
      qianhou.write(90);
      zuoyou.write(70);
      delay(10);
    }
    if (read_byte == '+'){
      qianhou_offset += 5;
      Serial.println(qianhou_offset);
    } 
    if (read_byte == '-'){
      qianhou_offset -= 5;
      Serial.println(qianhou_offset);
    } 
    if (read_byte == '1'){
      zuoyou_offset += 5;
      Serial.println(zuoyou_offset);
    } 
    if (read_byte == '2'){
      zuoyou_offset -= 5;
      Serial.println(zuoyou_offset);
    } 
  }
}
