import cv2
import numpy as np
import argparse
import random
import time
import serial

# シリアル通信の設定
ser = serial.Serial('/dev/ttyUSB0', 115200) #'/dev/ttyUSB0'に115200の速度で接続しserという名のシリアル通信オブジェクトを作成


# 人工知能モデルへ入力する画像の調整パラメータ：人工知能モデルに画像を供給する前に画像のサイズを調整するためのパラメータを設定しています
IN_WIDTH = 300   #横幅を300ピクセルに設定
IN_HEIGHT = 300  #縦幅を300ピクセルに設定

# Mobilenet SSD COCO学習済モデルのラベル一覧の定義
CLASS_LABELS = {0: 'background',
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
                5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
                10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
                14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
                23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
                38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
                57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
                65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
                73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
                81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                90: 'toothbrush'}

# 検出したいクラスのID定義
person_class_id = 1
bicycle_class_id = 2
car_class_id = 3
motorcycle_class_id = 4

# 引数（コマンドラインのオプション指定）の定義
ap = argparse.ArgumentParser()     #ArgumentParserをapに省略
ap.add_argument('-p', '--pbtxt', required=True,  #p pbtxt(構造)ファイルへのパスを指定, オプションでpbtxtファイルがTrue(必須)です。helpはhelp検索時に表示
                help='path to pbtxt file')
ap.add_argument('-w', '--weights', required=True, #w(重み) TensorFlow推論用のモデルの重みファイルへのファイルパスを指定するためのもの 
                help='path to TensorFlow inference graph')
ap.add_argument('-c', '--confidence', type=float, default=0.3,  #閾値のデフォルトを調整/小数点の値を指定
                help='minimum probability')
args = vars(ap.parse_args()) #argparse ライブラリを使用してコマンドライン引数を解析し、辞書形式でその値を args に格納しています
                             #この方法を使用することで、コマンドライン引数からの情報を簡単に利用できるため、スクリプトの柔軟性と拡張性が向上します。

colors = {}                  #colors = {} は空の辞書を作成しています。↓で枠カラーに使います。
# ラベル毎の枠色をランダムにセット
random.seed()                                #乱数生成器のシードを初期化します。これにより、毎回同じランダムな色が生成されることが保証されます。
for key in CLASS_LABELS.keys():              #CLASS_LABELS ディクショナリ内の各クラス(1~90)のラベルに対して繰り返し処理を行います。
    colors[key] = (random.randrange(255),    #指定されたクラスのラベル（key）にランダムな RGB 色を割り当てています。
                   random.randrange(255),    #  random.randrange(255) は 0 から 254 のランダムな整数を生成し
                   random.randrange(255))    #  RGB の各チャンネル（赤、緑、青）に対応します。

# 人工知能モデルの読み込み
try:                                                        #ブロック内のコードは、例外が発生する可能性がある操作を含んでいます。これにより、エラーが発生した場合でもプログラムがクラッシュするのを防ぎます。
    print('モデル読み込み...')
    net = cv2.dnn.readNet(args['weights'], args['pbtxt'])   #OpenCVの cv2.dnn モジュールを使用して、指定された重みファイルと構造ファイルを読み込んで人工知能モデルを構築します。
except cv2.error as e:                                      #モデル読み込み中にエラーが発生した場合に実行されるブロックです。エラーが発生した場合、エラーメッセージが表示されます。
    print(f'Error while loading the model: {e}')
    ser.close()                                             #シリアル通信をクローズし、リソースを解放します。
    exit(1)                                                 #プログラムを終了します。エラーが発生した場合、プログラムを正常に終了できないことを示します。いわゆる強制終了
    

# ビデオカメラ開始
try:                              
    print('ビデオカメラ開始...')
    cap = cv2.VideoCapture(0)                                #OpenCVの cv2.VideoCapture クラスを使用して、デフォルトのビデオカメラ（通常はカメラデバイス番号0）を開きます。この cap オブジェクトを使用して、カメラからのフレームを読み取ります。
    if not cap.isOpened():                                   #ビデオカメラが正常に開かれなかった場合をチェックします。
        raise RuntimeError("Could not open video device.")   #カメラが正常に開かれていない場合、指定されたエラーメッセージを表示し、RuntimeError 例外を発生させます。
except cv2.error as e:                                       #この行含む4行は(人工知能モデルの読み込み)時と同じ処理

    print(f'Error while starting the camera: {e}')
    ser.close()
    exit(1)

# OpenCVのチックメータ（ストップウォッチ）機能をtmという名前で使えるようにする
tm = cv2.TickMeter()                              #モデルの実行時間を正確に測定し、パフォーマンスのボトルネックを見つけるのに役立つツールです。実行時間が長すぎる場合、ハードウェアのアップグレードやモデルの軽量化を検討することができます

# 前のフレームの検出結果を保存する変数を初期化
previous_object_detected = False                  #previous_object_detected変数は物体の検出状態を追跡し、その変更を処理するためのトリガーとして機能します。これにより、物体の検出に関連するアクションを適切に制御できます。
# カメラを起動したときに検知 OFF 状態とする
object_detected = False                           #今回はOFFからONのA信号を確実に検知したい為OFFでスタートしています。

try:     #このコードブロックは、ビデオカメラからの画像の取得を試行し、取得に成功した場合は無限ループ内でその画像を処理するためのものです。
    while True:
        # カメラからの画像を読み込む
        ret, frame = cap.read()    #cap.read()を使ってビデオカメラからの新しいフレームを読み込みます。
        if not ret:          #ret はブール値で、True の場合は新しいフレームが正常に読み込まれ、False の場合は新しいフレームの読み込みに問題があることを示します
            break            #if not ret は、ret が False である場合にループを抜けるための条件です。新しいフレームが正常に読み込まれなかった場合、ループを終了し、プログラムが終了します。

        # 高さと幅情報を画像フレームから取り出す(この部分のコードは、ビデオフレームの高さと幅を取得しています。具体的には以下のことを行っています)
        (frame_height, frame_width) = frame.shape[:2]  #ビデオフレームのサイズ情報を含む NumPy 配列です.この配列から最初の2つの要素（高さと幅）を抽出します
                                                       #()内は高さと幅の値を代入します。これにより、後続のコードでフレームの高さと幅を簡単に参照できるようになります

       # 画像フレームを調整しblob形式(人工知能モデルへの入力形式)へ変換
        blob = cv2.dnn.blobFromImage(frame, size=(IN_WIDTH, IN_HEIGHT), swapRB=False, crop=False)
                
                #cv2.dnn.blobFromImage() 関数は、与えられた画像フレーム frame をモデルに入力できるように調整します。
                #size=(IN_WIDTH, IN_HEIGHT) により、モデルの期待する画像のサイズにリサイズします。13.14行で各300ピクセルと指定済み。
                #swapRB=False は、赤と青のチャンネルを入れ替えないように指定しています。多くのモデルは通常、OpenCV のデフォルトの色チャンネル順序（BGR）を期待しているため、これを無効にしておく必要があります。
                #crop=False は、画像の中央からトリミングしないように指定します。
                
        # blob形式の入力画像を人工知能にセット
        net.setInput(blob)   #↑のブロックで設定したモノをAIモデルにセット
        
        # 画像を人工知能へ流す
        tm.reset()                      #OpenCVのチックメータ（ストップウォッチ）をリセットします。これにより、時間の計測が新たに開始されます。
        tm.start()                      #チックメータの計測を開始します。この点から、物体検出の処理時間を計測し始めます。
        detections = net.forward()      #事前に読み込んだ人工知能モデル net に対して、入力画像 blob から物体検出を行う命令です。この行が実行されると、モデルは画像から物体を検出し、それらの検出結果を detections に格納します。
        tm.stop()                       #チックメータの計測を終了します.
        
        # 検出された物体の種別フラグを初期化
        person_detected = False
        bicycle_detected = False
        car_detected = False            #これらのフラグは、後で検出された物体の種別に関する情報を格納し、処理の制御や通知に使用される
        motorcycle_detected = False     #今回、特定の種別の物体が検出されたかどうかを追跡し、それに基づいてアクションを実行する為
        
        
         #このブロックから ～座標の取得のブロックまでのブロックは物体検出モデル（MobileNet SSD）を使用して、画像またはビデオフレームから検出されたオブジェクトに関する情報を処理しています。
         # 検出数（mobilenet SSDでは100）を繰り返す
        for i in range(detections.shape[2]):
            # i番目の検出オブジェクトの正答率を取り出す
            confidence = detections[0, 0, i, 2]

            # 正答率がしきい値を下回ったら何もしない(今回は30％以下)
            if confidence < args['confidence']:
                continue

            # 検出物体の種別と座標を取得
            class_id = int(detections[0, 0, i, 1])   #：この行では、detections 配列から特定の検出結果のクラスID（クラスラベル）を取得しています。detections[0, 0, i, 1] は、i 番目の検出結果に対するクラスIDを表します。

            if class_id == person_class_id:           #特定の物体クラス（person, bicycle, car, motorcycle）が検出されたかどうかを確認します.
                person_detected = True                #一致する場合に対応するフラグ変数を True に設定、特定の物体クラスが検出された事になります。
            elif class_id == bicycle_class_id:
                bicycle_detected = True
            elif class_id == car_class_id:
                car_detected = True
            elif class_id == motorcycle_class_id:
                motorcycle_detected = True

            # 枠をフレームに描画(各検出結果からバウンディングボックスの座標を取得し、OpenCVを使用してフレームに四角形を描画します。)
            
            #のコードの結果として、元のフレーム上に検出されたオブジェクトのバウンディングボックスが描画され、オブジェクトの位置が視覚的に表示されます。これは物体検出の結果を視覚化するため
            #start_x, start_y, end_x, end_y は、バウンディングボックスの四隅の座標を表す整数値です。これらの座標は、検出されたオブジェクトの位置を示します。
            
            start_x = int(detections[0, 0, i, 3] * frame_width)   #検出結果から、バウンディングボックスの左上隅のX座標を計算します。detections[0, 0, i, 3] は検出結果におけるバウンディングボックス左上のX座標の相対位置を示し、frame_width は元のフレームの幅です。
            start_y = int(detections[0, 0, i, 4] * frame_height)  #検出結果から、バウンディングボックスの左上隅のY座標を計算します。detections[0, 0, i, 4] は検出結果におけるバウンディングボックス左上のY座標の相対位置を示し、frame_height は元のフレームの高さです。
            end_x = int(detections[0, 0, i, 5] * frame_width)     #検出結果から、バウンディングボックスの右下隅のX座標を計算します。detections[0, 0, i, 5] は検出結果におけるバウンディングボックス右下のX座標の相対位置を示し、frame_width は元のフレームの幅です。
            end_y = int(detections[0, 0, i, 6] * frame_height)    #検出結果から、バウンディングボックスの右下隅のY座標を計算します。detections[0, 0, i, 6] は検出結果におけるバウンディングボックス右下のY座標の相対位置を示し、frame_height は元のフレームの高さです。
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors[class_id], 2)
            #この行では、OpenCVを使用して元のフレーム上にバウンディングボックスを描画します。具体的には、frame は描画されるフレーム、(start_x, start_y) はバウンディングボックスの左上隅の座標、
            #(end_x, end_y) はバウンディングボックスの右下隅の座標を指定します。また、colors[class_id] はバウンディングボックスの色を示し、2 は線の太さを表します。
            #ボックスの色は、物体のクラスによって異なる場合があるため、class_id を使用して対応する色を選択します。
            
            
            
            # 物体の種別を示す person といったラベルと確信度を label にセット
            label = CLASS_LABELS[class_id]               #検出された物体のクラス（種別）を取得します。これにより検出された物体のクラスが label 変数に格納されます。
            label += ': ' + str(round(confidence * 100, 2)) + '%'      #この行では、label に確信度（正答率）の情報を追加します。確信度は confidence 変数に格納されており、0から1の範囲の浮動小数点数として表されます。 str(round(confidence * 100, 2)) は、確信度を小数点以下2桁まで四捨五入して文字列に変換する操作です。
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)    #cv2.getTextSize 関数を使用して、テキストラベルのサイズを計算します。
                                                                                                #cv2.FONT_HERSHEY_SIMPLEX は使用されるフォント、0.5 はフォントのスケール、1 はフォントの太さを指定しています。計算された label_size はテキストラベルのサイズを表し、base_line はベースラインの位置を示します
            cv2.rectangle(frame, (start_x, start_y - label_size[1]), (start_x + label_size[0], start_y + base_line), (255, 255, 255), cv2.FILLED)   #この行は、テキストラベルの背景として白い矩形を描画します。(255, 255, 255) は白の色を表し、cv2.FILLED は矩形を塗りつぶす.x,y等は↑と同じく座標を表す
            cv2.putText(frame, label, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))  #frame：テキストを描画する対象のフレーム。label：描画するテキストラベルの内容。(start_x, start_y)：テキストラベルの左上隅の座標。cv2.FONT_HERSHEY_SIMPLEX：使用するフォント。0.5：フォントのスケール。(0, 0, 0)：テキストの色（ここでは黒）
      #フレームにAI処理にかかる時間を表示するための処理を行っています
        ai_time = tm.getTimeMilli()   #AI処理にかかる時間をミリ秒単位で計測し、その時間を ai_time 変数に格納します。
        cv2.putText(frame, '{:.2f} ms'.format(ai_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
          #計測されたAI処理の時間をフレーム上に表示するための処理です。
          #frame：テキストを描画する対象のフレーム。
          #'{:.2f} ms'.format(ai_time)：テキストの内容。ここでは ai_time の値を小数点以下2桁まで表示し、" ms"（ミリ秒）を追加しています。
          #(10, 30)：テキストの表示位置。テキストがフレームの左上に表示されます。
          #cv2.FONT_HERSHEY_SIMPLEX：使用するフォント。
          #1.0：フォントのスケール。
          #(0, 255, 0)：テキストの色。ここでは緑色です。
          #thickness=2：テキストの太さ。テキストは太く表示されます。



        # 検知が OFF から ON に切り替わった場合に"A"を送信
        if not object_detected and (person_detected or bicycle_detected or car_detected or motorcycle_detected):  この条件は、物体の検知がOFFからON(特定のモノ)に切り替わった場合を検出します。
            ser.write(b'A')
            object_detected = True    #object_detected フラグ変数を True に設定します。これにより、検知がONであることを示します。
            print("DETECTED")
        # 検知が ON から OFF に切り替わった場合に"UNDETECTED"と表示
        elif object_detected and not (person_detected or bicycle_detected or car_detected or motorcycle_detected):
            #ser.write(b'B')
            object_detected = False
            print("UNDETECTED")
            
        # このコードは、OpenCVを使用してカメラのライブ映像（ビデオフィード）をウィンドウに表示し、キーボードからの入力を待つためのループを処理しています。
        cv2.imshow('Live', frame)   #この行は、ウィンドウにフレームを表示するための処理です。'Live' はウィンドウのタイトルです。frame は表示するフレームの内容を含む画像です。

        if cv2.waitKey(1) >= 0:     #この行は、ユーザーからのキーボード入力を待ちます。cv2.waitKey(1) は、ユーザーがキーボードから入力を行うまでの待機時間をミリ秒単位で指定します。1ミリ秒ごとに入力をチェックします
            break                   #ループから抜け出します。つまり、キーボード入力を受け付けたらプログラムが終了します。

except KeyboardInterrupt:           #KeyboardInterrupt例外が発生した場合に実行される例外処理ブロックの開始を示します。通常、プログラムが実行中にユーザーがCtrl+Cを押すことによって発生します。この例外は、プログラムを中断または終了するために使用されます。
    pass                            #例外が発生した場合に実行する処理がないことを示しています
              
# 終了処理
ser.close()               #シリアルポート（通信ポート）を閉じるための処理です
cap.release()             #カメラキャプチャデバイス（ビデオカメラなど）を解放する処理です
cv2.destroyAllWindows()   #この行は、OpenCVウィンドウを閉じるための処理です 
print("blink end")        #プログラムが正常に終了したことを示すメッセージをコンソールに出力します。

