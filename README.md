# 🔌 การต่อสายระบบ Tomato Harvesting Robot

## 📋 รายการอุปกรณ์ที่ต้องใช้

| อุปกรณ์ | จำนวน | หมายเหตุ |
|---------|-------|----------|
| Raspberry Pi 4 | 4 | ติดตั้ง PyQt6 |
| Arduino Uno/Nano | 4 | ควบคุม Servo |
| Servo Motor | 20 | 5 ตัว/แขนกล |
| Breadboard | 4 | ขนาดใหญ่ |
| Power Supply 5V/5A | 4 | สำหรับ Servo |
| Pi Camera v2/v3 | 4 | หรือ USB Camera |

## 🔗 การเชื่อมต่อหลัก

### Raspberry Pi → Arduino
```
Raspberry Pi USB Port → Arduino USB Port
```

### Arduino → Servo Motors
```
Arduino D9  → Servo Base (ฐานหมุน)
Arduino D6  → Servo Shoulder (ไหล่)
Arduino D5  → Servo Elbow (ข้อศอก)
Arduino D3  → Servo Wrist (ข้อมือ)
Arduino D10 → Servo Gripper (ที่จับ)
```

### Power Connections
```
Power Supply 5V (+) → Breadboard Power Rail (Red)
Power Supply GND (-) → Breadboard Ground Rail (Black)
Arduino GND → Breadboard Ground Rail
All Servo Red Wires → Breadboard Power Rail
All Servo Brown Wires → Breadboard Ground Rail
```

### Camera Connection
```
Pi Camera Ribbon Cable → Raspberry Pi Camera CSI Port
หรือ
USB Camera → Raspberry Pi USB Port
```

## ⚡ สรุปการต่อสาย Servo

```
Servo Red Wire (Power) → Breadboard Power Rail (5V)
Servo Brown Wire (Ground) → Breadboard Ground Rail
Servo Orange Wire (Signal) → Arduino Pin (ตามตาราง)
```

## 💻 Serial Port Mapping

```
USB Port 1 → /dev/ttyUSB0 → Robot Arm 0
USB Port 2 → /dev/ttyUSB1 → Robot Arm 1
USB Port 3 → /dev/ttyUSB2 → Robot Arm 2
USB Port 4 → /dev/ttyUSB3 → Robot Arm 3
```

## 📍 Pin Assignment Table

```
┌─────────────┬──────────────┬─────────────┐
│ Arduino Pin │ Servo Number │   Function  │
├─────────────┼──────────────┼─────────────┤
│     D9      │   Servo 1    │    Base     │
│     D6      │   Servo 2    │  Shoulder   │
│     D5      │   Servo 3    │    Elbow    │
│     D3      │   Servo 4    │    Wrist    │
│     D10     │   Servo 5    │   Gripper   │
└─────────────┴──────────────┴─────────────┘
```

## 🔧 Breadboard Layout

```
Breadboard Power Rail (+5V)  ══════════════════════
                              │ │ │ │ │
                              S S S S S (Servo Power)
                              
Breadboard Ground Rail (GND) ══════════════════════
                              │ │ │ │ │
                              S S S S S (Servo Ground)
```

## ✅ Checklist การต่อสาย

```
□ Raspberry Pi ต่อกับ Arduino ผ่าน USB
□ Arduino GND ต่อกับ Breadboard Ground Rail
□ Power Supply 5V ต่อกับ Breadboard Rails
□ Servo ทั้ง 5 ตัวต่อกับ Arduino Pins ที่กำหนด
□ Pi Camera ต่อกับ CSI Port
□ ตรวจสอบ Serial Port ที่ /dev/ttyUSB*
```

## 🚀 Quick Start Commands

```bash
# 1. ต่อ Hardware ตามตาราง



# 2. เริ่มใช้งาน
python3 main.py
```

## ⚠️ ข้อควรระวัง

```
! ห้ามต่อ Servo กับ Raspberry Pi GPIO โดยตรง (3.3V)
! ต้องใช้ Power Supply แยกสำหรับ Servo
! ต้องต่อ Ground ร่วมกันระหว่าง Arduino และ Power Supply
```
