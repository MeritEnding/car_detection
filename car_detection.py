import cv2
import numpy as np

# YOLO 모델 설정 파일과 가중치 파일의 경로
yolo_config_path = "yolov3.cfg"  # 설정 파일 경로
yolo_weights_path = "yolov3.weights"  # 가중치 파일 경로
yolo_classes_path = "coco.names"  # 클래스 이름 파일 경로

# YOLO 모델 초기화
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# 클래스 이름 로드
with open(yolo_classes_path, 'r') as f:
    classes = f.read().strip().split('\n')
# 이미지 읽기
image = cv2.imread("car.jpg")  # 차량을 포함한 이미지 파일 경로

# YOLO 모델 입력 크기 설정 (보통 416x416 사용)
input_size = (416, 416)

# 이미지를 YOLO 모델에 맞게 전처리
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=input_size, swapRB=True, crop=False)

# YOLO 모델에 전처리된 이미지 입력
net.setInput(blob)

# YOLO 모델 출력 계산
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

# 객체 감지 결과 처리
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # 탐지 신뢰도 조건 (일반적으로 0.5 이상인 경우만 인식)
        if confidence > 0.5:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])

            # 경계 상자 좌표 계산
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            # 경계 상자 그리기
            color = (0, 255, 0)  # 초록색 경계 상자
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 결과 이미지 저장
cv2.imwrite("output.jpg", image)

# 결과 이미지 표시
cv2.imshow("Vehicle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
