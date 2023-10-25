//ここまでピンの設定
#define InputPin 32
#define echoPin 33
#define trigPin 17
#define bzPIN 13
#define ledPin 26
#define on HIGH
#define off LOW


//音階の周波数
#define DO 261.6
#define MI 329.63


//wifi,http呼び出し機能のインストール
#include <WiFi.h>
#include <HTTPClient.h>


//wifiのSSIDとpassの設定
const char* ssid = "AP01-01";
const char* password = "1qaz2wsx";


const String endpoint = "https://maker.ifttt.com/trigger/"; // IFTTTのエンドポイントURL
const String eventName = "SchooMyIoT"; // IFTTTのイベント名
const String conn = "/with/key/";
const String Id = "jX0-XczR5iRUlvn0m2uDaUdoi5nu0iN1-TI9cf9iVPk"; // あなたのIFTTTのキー
const String value = "?value1="; // パラメータの値を指定（必要に応じてvalue2, value3を追加）


//型の設定
double Duration = 0;
double Distance = 0;
byte val;
int inp;
bool isFirstSignalReceived = false;


//音楽内容
void playmusic() {
    ledcWriteTone(0, MI);
    delay(500);
    ledcWriteTone(0, DO);
    delay(500);
    ledcWriteTone(0, 0); // no sound
    delay(100);

}

bool isPlaying = false;

//音楽停止
void stopMusic() {
    ledcWriteTone(0, 0); // no sound
    delay(100);
    isPlaying = false;

}

//各ピンの初期状態
void setup() {
    pinMode(ledPin, OUTPUT);
    pinMode(ledPin, on);
    pinMode(echoPin, INPUT);
    pinMode(trigPin, OUTPUT);
    ledcSetup(0, 12000, 8);
    ledcAttachPin(bzPIN, 0);
    pinMode(InputPin, INPUT);
    Serial.begin(115200);
。

// WiFi接続
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");
}

//衝撃検知センサーのセット
int SensAve(){
  int i;
  int sval=0;
  int ave=0;
  for (i = 0; i < 10000; i++){
    sval = sval + digitalRead(InputPin);
  }
  ave = sval / 100;
  return ave;

}

void loop() {
  {
//衝撃検知センサー起動
  inp = SensAve();
  Serial.println(inp);
  if (inp >= 80 && inp <= 99) { 
    sendIFTTTMessage();

  }else{}
  delay(100); 
}

{
    if (Serial.available() > 0) {//ラズパイ側から送られてきている信号の受け取り。
        val = Serial.read();//変数valにラズパイからの信号を代入
        if (val == 'A' && !isFirstSignalReceived) {
        isFirstSignalReceived = true;
        }//変数valが'A'と同値であり、!isFirstSignalReceivedだった場合以下プログラムの開始
        if (isFirstSignalReceived && Serial.available() > 0) {
            while (Serial.available() > 0) {
            char dummy = Serial.read();
            }
            Serial.flush();
            delay(1000);
            //2回目以降isFirstSignalReceivedが入ってきた場合A信号残留無し。
        } 
        while(true){
            //距離センサーのループ
            Serial.println("ラズパイ検知");
            digitalWrite(trigPin, LOW);
            delayMicroseconds(2);
            digitalWrite(trigPin, HIGH);
            delayMicroseconds(10);
            digitalWrite(trigPin, LOW);
            Duration = pulseIn(echoPin, HIGH);
            //距離センサーの検知開始

            if (Duration > 0) {
                Duration = Duration / 2;
                Distance = Duration * 340 * 100 / 1000000;
                Serial.print("Distance:");
                Serial.print(Distance);
                Serial.println("cm");
                //検知した距離を片道のcmに直す。

                if (Distance >= 20 && Distance <= 40) {
                    digitalWrite(ledPin, on);
                    playmusic();
                    digitalWrite(ledPin, off);
                    delay(1000); 
                    stopMusic();
                    isPlaying = true;
                    break;
                    //検知距離が20～40cm以内だった場合振動・音楽再生
                    //breakにより抜け出し
                } else {
                    digitalWrite(ledPin, on);
                    stopMusic();
                    isPlaying = true;
                    //検知外だった場合何もなしで距離検知に戻る
                }
            }
            digitalWrite(ledPin, on);
            delay(100);
            }
            
            } else {
            Serial.println("ラズパイ検知なし");
            digitalWrite(ledPin, on);
            stopMusic();
            isPlaying = true;
            //ラズパイからの信号が何もなかった場合何もしない。
        }
    }
}


void sendIFTTTMessage() {
   if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(endpoint + eventName + conn + Id + value); //URLを指定

    http.addHeader("Content-Type", "application/x-www-form-urlencoded"); //必要なヘッダーを追加
    int httpCode = http.GET();  //GETリクエストを送信

    if (httpCode == 200) { //返答がある場合
      Serial.println("IFTTT Message Sent: 200 OK");
    } else {
      Serial.println("Error on HTTP request");
    }
    http.end(); //Free the resources
  }
}
