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
#define BYTES 1024
#define ZERO_ON_EXIT 0

unsigned char m[7] = {10, 90, 45, 180, 180, 90, 10};
unsigned char stepdelay = 10;
unsigned char read_buf[BYTES] = {10, 90, 45, 180 ,180, 90, 10};
unsigned char nbytes = 0;
unsigned char i = 0;
  
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200 ); //initialize serial COM at 9600 baudrate
  pinMode(LED_BUILTIN, OUTPUT); //make the LED pin (13) as output
  digitalWrite (LED_BUILTIN, LOW);
  Serial.println("Hi!, I am Arduino");
  Braccio.begin();
}
 
void loop() {
  // put your main code here, to run repeatedly:
  
  while (Serial.available()){
    if((read_buf[nbytes] = Serial.read()) != 0) nbytes++;
     }

  while ( i < nbytes){
     m[i] = read_buf[i];
     i++;
  }

  Braccio.ServoMovement(m[0], m[1], m[2], m[3], m[4], m[5], m[6]);

  if( Serial.availableForWrite()>= sizeof(m) ){
    Serial.write(m, sizeof(m));
  }
   
  
  
}
