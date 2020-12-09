#include <Servo.h>
#include <Braccio.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

#define base_min 0
#define base_max 180
#define shoulder_min 15
#define shoulder_max 165
#define elbow_min 0
#define elbow_max 180
#define wristv_min 0
#define wristv_max 180
#define wristr_min 0
#define wristr_max 180
#define gripper_min 10
#define gripper_max 73
#define WRITE_BYTES 7
#define READ_BYTES WRITE_BYTES
#define ZERO_ON_EXIT 0

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); //initialize serial COM at 9600 baudrate
  pinMode(LED_BUILTIN, OUTPUT); //make the LED pin (13) as output
  digitalWrite (LED_BUILTIN, LOW);
  Serial.println("Hi!, I am Arduino");
  Braccio.begin();
}
unsigned char m1 = 90;   // Base
unsigned char m2 = 45;   // Shoulder
unsigned char m3 = 180;  // Elbow
unsigned char m4 = 180;  // WristV
unsigned char m5 = 90;   // WristR
unsigned char m6 = 10;   // Grabber
unsigned char stepdelay = 10;
//unsigned char write_buf[WRITE_BYTES] = {0};
unsigned char read_buf[WRITE_BYTES] = {10, 90, 45, 180 ,180, 90, 10};
char bytes_written = 0;
 
void loop() {
  // put your main code here, to run repeatedly:

  
  if (Serial.available() == 7){
    for(int i = 0; i < 7; i++)
    {
      if((read_buf[i] = Serial.read()) == -1)break;
    }
   
  }


  stepdelay = read_buf[0];
  m1 = read_buf[1];
  m2 = read_buf[2];
  m3 = read_buf[3];
  m4 = read_buf[4];
  m5 = read_buf[5];
  m6 = read_buf[6];

  Braccio.ServoMovement(stepdelay, m1, m2, m3, m4, m5, m6);

//  write_buf[0] = stepdelay;
//  write_buf[1] = m1;
//  write_buf[2] = m2;
//  write_buf[3] = m3;
//  write_buf[4] = m4;
//  write_buf[5] = m5;
//  write_buf[6] = m6;
//  
//  
//  if(Serial.availableForWrite() > WRITE_BYTES){
//    do{
//      bytes_written += Serial.write(write_buf, WRITE_BYTES);
//    }while(bytes_written < WRITE_BYTES);
//
//    bytes_written = ZERO_ON_EXIT;
//  }
//  
  
  
  
}
