import cv2
import mediapipe as mp
import numpy as np
import time, os

#차례로 0 1 2 3 4 5 6
actions = ['hook_raise', 'hook_down']  #액션의 종류는 7개
seq_length = 30 #윈도우 크기, 30개씩 쪼개서(LSTM 이기 때문에 사이즈 지정)
secs_for_action = 30 # 액션을 녹화하는 시간, 학습이 잘 안될 시 변수 크기 조정
#30초씩 총 180초 녹화함 (액션 6개)
#녹화 중 멈추는 프레임에 다음 액션 준비


# MediaPipe hands model
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0) #비디오 캡쳐로 웹캡인식

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) #dataset 폴더 생성

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read() #웹캡의 첫번째 이미지 불러오기

        img = cv2.flip(img, 1) #좌우 대칭flip

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000) #3초 대기후 다시 녹화

        start_time = time.time()

        while time.time() - start_time < secs_for_action: #지정한 시간 녹화 (30초동안 반복)
            ret, img = cap.read()

            img = cv2.flip(img, 1) #frame을 읽어 Mediapipe에 넣어줌
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] #visibility -> 손가락의 노드가 웹캠에 보는지 여부

                    # 손가락 사이의 각도를 구하는 코드
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    
                    
                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) #joint사이의 각도를 구하는 코드 입력

# 여기까지
                    angle_label = np.array([angle], dtype=np.float32) #action label 생성
                    angle_label = np.append(angle_label, idx) #인덱스 0부터 6까지 라벨 부여

                    d = np.concatenate([joint.flatten(), angle_label]) #연결 (100개의 사이즈의 행렬)

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)  #모은 데이터를 넘파이 배열로 만듬    
        print(action, data.shape) #모은 데이터 차원 출력
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # 데이터 셋 npy형태로 저장
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data) #dataset에 저장
    
    # raw_data, sequence_data 로 저장함 (총 14개의 파일 생성)
    # sequence_data를 사용해 학습을 진행함
    break


