"""
================================================================================
КОД СОЗДАН НА ОСНОВЕ СЛЕДУЮЩИХ ПРОМПТОВ (ЗАПРОСОВ ПОЛЬЗОВАТЕЛЯ):
================================================================================

[Промпт 1]
"Создай нейросеть на YOLO для манипулятора с камерой, нейросеть должна определить
что за предмет и узнать расстояние до предмета и взять предмет и положить в корзину."
=> Результат: общая архитектура системы и выбор YOLOv8.

[Промпт 2]
"Как YOLOv8 может определять расстояние через обычную камеру"
=> Результат: обоснование и выбор метода подобных треугольников (Similar Triangles).

[Промпт 3]
"Создай код для манипулятора с обычной камерой используя метод Треугольник подобия
найди расстояние до объекта когда камера повернута вниз и что за объект и как его хватать"
=> Результат: настоящий скрипт с учётом наклона камеры и логикой захвата.

================================================================================
"""

import cv2
import math
import numpy as np
from ultralytics import YOLO

# ============================================================
# 1. НАСТРОЙКИ И КАЛИБРОВОЧНЫЕ ПАРАМЕТРЫ
# ============================================================
# (Ответ на промпт: "камера повернута вниз" и "метод Треугольник подобия")

# --- Калибровка камеры ---
# Значение FOCAL_LENGTH_PX должно быть определено экспериментально.
# Подробнее в ответе на промпт: "Как YOLOv8 может определять расстояние..."
FOCAL_LENGTH_PX = 800.0   # Пример для веб-камеры 640x480

# Реальные размеры объектов (в метрах) – необходимы для метода подобных треугольников.
# В промышленном решении эти значения берутся из базы данных или 3D-модели.
KNOWN_WIDTH = {
    'cube': 0.05,        # кубик со стороной 5 см
    'apple': 0.08,       # яблоко диаметром 8 см
    'bottle': 0.07,      # бутылка диаметром 7 см
    'default': 0.06      # значение по умолчанию
}

# --- Геометрия установки камеры (Ответ на промпт: "камера повернута вниз") ---
CAMERA_HEIGHT = 0.5          # Высота камеры над столом, м
CAMERA_TILT_DEG = 15.0       # Угол наклона камеры от вертикали (0 = строго вниз)
CAMERA_TILT_RAD = math.radians(CAMERA_TILT_DEG)

# Разрешение кадра
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH / 2
CENTER_Y = FRAME_HEIGHT / 2

# Угловое разрешение (радиан на пиксель) – упрощённая модель.
ANGLE_PER_PIXEL = math.radians(60) / FRAME_WIDTH

# ============================================================
# 2. ЗАГРУЗКА МОДЕЛИ YOLO (Ответ на промпт: "нейросеть на YOLO")
# ============================================================
model = YOLO('yolov8n.pt')   # используйте свою обученную модель

# ============================================================
# 3. ЗАГЛУШКА ДЛЯ МАНИПУЛЯТОРА (ЗАМЕНИТЕ НА РЕАЛЬНОЕ API)
# ============================================================
# (Ответ на промпт: "взять предмет и положить в корзину")
class MockRobotArm:
    def __init__(self):
        print("Манипулятор инициализирован.")
        self.home = [0, 0, 0, 0, 0, 0]

    def move_to(self, x, y, z, orientation='vertical'):
        """Перемещение схвата в точку (x,y,z) в системе координат манипулятора."""
        print(f"Перемещение в точку: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, ориентация={orientation}")

    def grasp(self, object_class):
        """Выполнение захвата в зависимости от класса объекта (Ответ на промпт: 'как его хватать')."""
        if object_class in ['cube', 'box']:
            print("Используется двухпальцевый захват.")
        elif object_class in ['bottle', 'cylinder']:
            print("Используется захват с обхватом.")
        else:
            print("Используется стандартный захват.")
        print("Захват выполнен.")

    def place(self, bin_position):
        """Перемещение объекта в корзину."""
        print(f"Перемещение в корзину: {bin_position}")

arm = MockRobotArm()

# ============================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def calculate_distance(bbox_width_px, real_width_m, focal_length_px):
    """
    [Промпт: "метод Треугольник подобия"]
    Расстояние от камеры до объекта вдоль оптической оси.
    Формула: D = (f * W_real) / w_px
    """
    if bbox_width_px == 0:
        return None
    return (focal_length_px * real_width_m) / bbox_width_px

def pixel_to_world(px, py, distance, tilt_rad, cam_height):
    """
    [Промпт: "камера повернута вниз"]
    Преобразование координат пикселя в мировые координаты (X, Y, Z)
    относительно основания камеры с учётом её наклона.
    """
    delta_x_rad = (px - CENTER_X) * ANGLE_PER_PIXEL
    delta_y_rad = (py - CENTER_Y) * ANGLE_PER_PIXEL

    dx = distance * math.tan(delta_x_rad)
    dy = distance * math.tan(delta_y_rad)

    cam_x = dx
    cam_y = distance
    cam_z = -dy

    # Поворот вокруг оси X на угол наклона камеры
    world_x = cam_x
    world_y = cam_y * math.cos(tilt_rad) + cam_z * math.sin(tilt_rad)
    world_z = cam_height - (cam_y * math.sin(tilt_rad) - cam_z * math.cos(tilt_rad))

    return world_x, world_y, world_z

def get_grasp_strategy(class_name):
    """
    [Промпт: "как его хватать"]
    Возвращает параметры захвата в зависимости от распознанного класса.
    """
    strategies = {
        'cube':   {'approach': 'top',    'gripper': 'parallel', 'pre_grasp_offset': 0.05},
        'bottle': {'approach': 'side',   'gripper': 'encompass', 'pre_grasp_offset': 0.08},
        'apple':  {'approach': 'top',    'gripper': 'soft',      'pre_grasp_offset': 0.03},
    }
    return strategies.get(class_name, {'approach': 'top', 'gripper': 'standard', 'pre_grasp_offset': 0.05})

# ============================================================
# 5. ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕО
# ============================================================
def main():
    """
    [Объединяет все промпты]:
    - YOLO для определения объекта
    - Метод подобия для расстояния
    - Учёт наклона камеры
    - Логика захвата и перемещения в корзину
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    print("Нажмите 'q' для выхода, 'g' для захвата первого обнаруженного объекта.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция YOLO
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if confidence < 0.5:
                    continue

                bbox_width = x2 - x1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                real_width = KNOWN_WIDTH.get(class_name, KNOWN_WIDTH['default'])
                distance = calculate_distance(bbox_width, real_width, FOCAL_LENGTH_PX)
                if distance is None or distance <= 0:
                    continue

                obj_x, obj_y, obj_z = pixel_to_world(center_x, center_y, distance,
                                                     CAMERA_TILT_RAD, CAMERA_HEIGHT)

                # В реальной системе сюда добавляется матрица hand-eye калибровки
                robot_x, robot_y, robot_z = obj_x, obj_y, obj_z

                label = f"{class_name} {confidence:.2f} Dist:{distance:.2f}m"
                cv2.putText(annotated_frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(annotated_frame, f"Pos: ({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f})",
                            (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('g'):
                    strategy = get_grasp_strategy(class_name)
                    pre_grasp_z = robot_z + strategy['pre_grasp_offset']
                    arm.move_to(robot_x, robot_y, pre_grasp_z, orientation='vertical')
                    arm.move_to(robot_x, robot_y, robot_z, orientation='vertical')
                    arm.grasp(class_name)
                    arm.move_to(robot_x, robot_y, pre_grasp_z, orientation='vertical')
                    bin_pos = [0.3, 0.2, 0.1]   # координаты корзины
                    arm.place(bin_pos)
                    print("Цикл захвата завершён.")
                    break

        cv2.imshow('YOLO + Distance Estimation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()