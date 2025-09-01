"""
Jogo de Damas controlado por gestos (punho/"pinça" com polegar + indicador) usando Webcam, OpenCV e MediaPipe Hands.

Requisitos (instale no terminal):
    python -m pip install --upgrade pip
    pip install opencv-python mediapipe numpy

Execução:
    python damas_gestos.py

Gestos:
  • "Cursor" = ponta do dedo indicador.
  • "Clique" = gesto de pinça (polegar e indicador se aproximam).
  • Selecione uma peça do seu turno com um "clique" sobre ela.
  • Depois, "clique" em uma casa válida destacada para mover.

Notas:
  • Implementa regras básicas: movimentos diagonais, capturas simples, coroação.
  • Não força captura obrigatória, e não implementa múltiplas capturas em cadeia (simplificado).
  • Há fallback por mouse: clique esquerdo seleciona/move.
  • Tecla 'r' reinicia a partida; 'q' sai.

Dicas Linux (webcam fechando):
  • Tente exportar:  export OPENCV_VIDEOIO_PRIORITY_MSMF=0  (no Windows) – ignorar no Linux.
  • No Linux/Ubuntu: o backend V4L2 é usado por padrão; este script também tenta CAP_V4L2.
  • Se tiver mais de uma câmera, mude CAMERA_INDEX abaixo.
"""

import cv2
import time
import math
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Mediapipe não encontrado. Instale com: pip install mediapipe")

# ===================== CONFIGURAÇÕES ===================== #
CAMERA_INDEX = 0
FRAME_W = 1280
FRAME_H = 720
BOARD_SIZE = 8  # 8x8 damas
MARGIN = 40     # margem interna ao redor do tabuleiro
PINCH_THRESHOLD = 0.035  # distância normalizada (0-1) entre indicador e polegar para considerar 'clique'
SMOOTHING = 0.35         # suavização do cursor (0=sem, 1=fixo)

# Cores (BGR)
COLOR_BG = (30, 30, 30)
COLOR_LIGHT = (220, 220, 220)
COLOR_DARK = (70, 70, 70)
COLOR_GRID = (40, 40, 40)
COLOR_HIGHLIGHT = (50, 180, 255)
COLOR_SELECTED = (0, 180, 120)
COLOR_VALID = (0, 200, 255)
COLOR_TEXT = (245, 245, 245)
COLOR_RED = (50, 50, 240)
COLOR_BLACK = (20, 20, 20)
COLOR_GHOST = (140, 140, 140)

# ===================== HAND TRACKER ===================== #
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self, max_num_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        self.last_cursor = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self.hands.process(rgb)
        cursor = None
        is_pinch = False
        if out.multi_hand_landmarks:
            # pegue a primeira mão detectada
            hand = out.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            # landmarks: índice 8 = ponta do indicador; 4 = ponta do polegar
            idx_tip = hand.landmark[8]
            thb_tip = hand.landmark[4]
            x = int(idx_tip.x * w)
            y = int(idx_tip.y * h)
            cursor = np.array([x, y], dtype=np.float32)
            # distância normalizada na imagem (aprox.)
            dx = idx_tip.x - thb_tip.x
            dy = idx_tip.y - thb_tip.y
            dist = math.hypot(dx, dy)
            is_pinch = dist < PINCH_THRESHOLD
        # suavização
        if cursor is not None:
            if self.last_cursor is None:
                self.last_cursor = cursor
            else:
                self.last_cursor = (1 - SMOOTHING) * cursor + SMOOTHING * self.last_cursor
        return (None if self.last_cursor is None else self.last_cursor.astype(int)), is_pinch

# ===================== LÓGICA DO TABULEIRO ===================== #
# Representação das peças:
# 0 = vazio
# 1 = vermelho (jogador 1) | 3 = vermelho dama
# 2 = preto   (jogador 2) | 4 = preto dama
RED, RED_K, BLK, BLK_K = 1, 3, 2, 4

class Checkers:
    def __init__(self, size=8):
        self.n = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.n, self.n), dtype=np.int32)
        # preencher peças (nas casas escuras) – 3 linhas para cada lado
        for y in range(3):
            for x in range(self.n):
                if (x + y) % 2 == 1:
                    self.board[y, x] = BLK
        for y in range(self.n - 3, self.n):
            for x in range(self.n):
                if (x + y) % 2 == 1:
                    self.board[y, x] = RED
        self.turn = RED  # vermelho começa
        self.selected = None  # (x, y)
        self.valid_moves = []  # lista de (to_x, to_y)
        self.winner = 0

    def inside(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n

    def piece_at(self, x, y):
        if not self.inside(x, y):
            return 0
        return self.board[y, x]

    def is_red(self, p):
        return p == RED or p == RED_K

    def is_black(self, p):
        return p == BLK or p == BLK_K

    def is_king(self, p):
        return p == RED_K or p == BLK_K

    def turn_is_red(self):
        return self.turn == RED

    def toggle_turn(self):
        self.turn = BLK if self.turn == RED else RED

    def gen_moves_for(self, x, y):
        p = self.piece_at(x, y)
        if p == 0:
            return []
        dirs = []
        if p == RED:
            dirs = [(-1, -1), (1, -1)]  # vermelho sobe
        elif p == BLK:
            dirs = [(-1, 1), (1, 1)]    # preto desce
        else:  # damas
            dirs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

        moves = []
        # passos simples
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self.inside(nx, ny) and self.piece_at(nx, ny) == 0:
                moves.append((nx, ny))
        # capturas simples
        for dx, dy in dirs:
            mx, my = x + dx, y + dy
            jx, jy = x + 2*dx, y + 2*dy
            if self.inside(jx, jy) and self.piece_at(jx, jy) == 0:
                mid = self.piece_at(mx, my)
                if mid != 0:
                    if self.is_red(p) and self.is_black(mid):
                        moves.append((jx, jy))
                    if self.is_black(p) and self.is_red(mid):
                        moves.append((jx, jy))
        return moves

    def select(self, x, y):
        p = self.piece_at(x, y)
        if p == 0:
            self.selected = None
            self.valid_moves = []
            return False
        if self.turn == RED and not self.is_red(p):
            return False
        if self.turn == BLK and not self.is_black(p):
            return False
        self.selected = (x, y)
        self.valid_moves = self.gen_moves_for(x, y)
        return True

    def move(self, to_x, to_y):
        if self.selected is None:
            return False
        from_x, from_y = self.selected
        if (to_x, to_y) not in self.valid_moves:
            return False
        p = self.piece_at(from_x, from_y)
        self.board[from_y, from_x] = 0
        # detectar captura
        if abs(to_x - from_x) == 2 and abs(to_y - from_y) == 2:
            mx = (from_x + to_x) // 2
            my = (from_y + to_y) // 2
            self.board[my, mx] = 0
        # coroação
        if p == RED and to_y == 0:
            p = RED_K
        elif p == BLK and to_y == self.n - 1:
            p = BLK_K
        self.board[to_y, to_x] = p
        self.selected = None
        self.valid_moves = []
        # checar vitória simples
        if not self.any_moves_left(for_player=BLK if self.turn == RED else RED):
            self.winner = self.turn
        self.toggle_turn()
        return True

    def any_moves_left(self, for_player):
        for y in range(self.n):
            for x in range(self.n):
                p = self.board[y, x]
                if p == 0:
                    continue
                if for_player == RED and not self.is_red(p):
                    continue
                if for_player == BLK and not self.is_black(p):
                    continue
                if self.gen_moves_for(x, y):
                    return True
        return False

# ===================== RENDERIZAÇÃO ===================== #
class Renderer:
    def __init__(self, width, height, board_n, margin):
        self.w = width
        self.h = height
        self.n = board_n
        self.margin = margin
        # cálculo da área do tabuleiro (quadrada, centralizada)
        side = min(self.w, self.h) - 2 * self.margin
        self.board_px = side
        self.cell = side // self.n
        self.top_left = ((self.w - side) // 2, (self.h - side) // 2)

    def to_cell(self, px, py):
        x0, y0 = self.top_left
        x = (px - x0) // self.cell
        y = (py - y0) // self.cell
        return int(x), int(y)

    def cell_center(self, cx, cy):
        x0, y0 = self.top_left
        cxp = int(x0 + cx * self.cell + self.cell/2)
        cyp = int(y0 + cy * self.cell + self.cell/2)
        return cxp, cyp

    def draw_board(self, img):
        x0, y0 = self.top_left
        # fundo
        cv2.rectangle(img, (x0-8, y0-8), (x0 + self.board_px + 8, y0 + self.board_px + 8), COLOR_GRID, 2)
        for y in range(self.n):
            for x in range(self.n):
                c = COLOR_LIGHT if (x + y) % 2 == 0 else COLOR_DARK
                x1 = x0 + x * self.cell
                y1 = y0 + y * self.cell
                cv2.rectangle(img, (x1, y1), (x1 + self.cell, y1 + self.cell), c, -1)

    def draw_pieces(self, img, game: Checkers):
        for y in range(self.n):
            for x in range(self.n):
                p = game.board[y, x]
                if p == 0:
                    continue
                cx, cy = self.cell_center(x, y)
                r = int(self.cell * 0.42)
                if p in (RED, RED_K):
                    color = COLOR_RED
                else:
                    color = COLOR_BLACK
                cv2.circle(img, (cx, cy), r, color, -1)
                cv2.circle(img, (cx, cy), r, (255, 255, 255), 2)
                if p in (RED_K, BLK_K):
                    cv2.circle(img, (cx, cy), int(r*0.6), (255, 215, 0), 3)

    def draw_selection(self, img, game: Checkers):
        if game.selected is not None:
            x, y = game.selected
            x0, y0 = self.top_left
            x1 = x0 + x * self.cell
            y1 = y0 + y * self.cell
            cv2.rectangle(img, (x1, y1), (x1 + self.cell, y1 + self.cell), COLOR_SELECTED, 3)
            # válidos
            for (vx, vy) in game.valid_moves:
                cx, cy = self.cell_center(vx, vy)
                cv2.circle(img, (cx, cy), int(self.cell*0.18), COLOR_VALID, -1)

    def draw_hud(self, img, fps, turn, pinch, winner):
        text_turn = 'Vermelho' if turn == RED else 'Preto'
        cv2.putText(img, f'FPS: {fps:.1f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(img, f'Turno: {text_turn}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(img, f'Clique: {"SIM" if pinch else "nao"}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)
        if winner:
            wtxt = 'Vermelho venceu!' if winner == RED else 'Preto venceu!'
            cv2.putText(img, wtxt, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 3, cv2.LINE_AA)

    def draw_cursor(self, img, cursor):
        if cursor is None:
            return
        x, y = int(cursor[0]), int(cursor[1])
        cv2.circle(img, (x, y), 10, COLOR_HIGHLIGHT, -1)
        cv2.circle(img, (x, y), 20, COLOR_HIGHLIGHT, 2)

# ===================== INTERAÇÃO ===================== #
class Controller:
    def __init__(self, renderer: Renderer, game: Checkers):
        self.r = renderer
        self.g = game
        self.last_pinch = False
        self.last_click_time = 0
        self.click_cooldown = 0.22  # seg, evita duplo clique

    def in_board(self, px, py):
        x0, y0 = self.r.top_left
        return x0 <= px < x0 + self.r.board_px and y0 <= py < y0 + self.r.board_px

    def handle_gesture(self, cursor, is_pinch):
        now = time.time()
        clicked = False
        if is_pinch and not self.last_pinch and (now - self.last_click_time) > self.click_cooldown:
            clicked = True
            self.last_click_time = now
        self.last_pinch = is_pinch

        if cursor is None:
            return
        px, py = int(cursor[0]), int(cursor[1])
        if not self.in_board(px, py):
            return
        cx, cy = self.r.to_cell(px, py)
        if not (0 <= cx < self.g.n and 0 <= cy < self.g.n):
            return
        if clicked:
            if self.g.selected is None:
                self.g.select(cx, cy)
            else:
                if not self.g.move(cx, cy):
                    # se clique inválido, tente selecionar uma nova peça
                    self.g.select(cx, cy)

    def handle_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.in_board(x, y):
                cx, cy = self.r.to_cell(x, y)
                if self.g.selected is None:
                    self.g.select(cx, cy)
                else:
                    if not self.g.move(cx, cy):
                        self.g.select(cx, cy)

# ===================== MAIN ===================== #

def open_camera(index=0, w=FRAME_W, h=FRAME_H):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("Não foi possível acessar a câmera. Verifique permissões e índice da câmera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    game = Checkers(BOARD_SIZE)
    renderer = Renderer(FRAME_W, FRAME_H, BOARD_SIZE, MARGIN)
    controller = Controller(renderer, game)

    tracker = HandTracker()

    cap = open_camera(CAMERA_INDEX, FRAME_W, FRAME_H)
    cv2.namedWindow('Damas por Gestos', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Damas por Gestos', controller.handle_mouse)

    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            # tente reabrir rapidamente
            cap.release()
            time.sleep(0.1)
            cap = open_camera(CAMERA_INDEX, FRAME_W, FRAME_H)
            ok, frame = cap.read()
            if not ok:
                break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame[:] = frame  # placeholder se quiser espelhar
        # opcional: espelhar para comportamento mais natural
        frame = cv2.flip(frame, 1)

        # fundo levemente escurecido
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_W, FRAME_H), COLOR_BG, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # mão -> cursor
        cursor, is_pinch = tracker.process(frame)

        # desenhar tabuleiro
        renderer.draw_board(frame)
        renderer.draw_pieces(frame, game)
        renderer.draw_selection(frame, game)

        # interações
        controller.handle_gesture(cursor, is_pinch)

        # cursor
        renderer.draw_cursor(frame, cursor)

        # FPS
        now = time.time()
        dt = now - prev
        prev = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
        renderer.draw_hud(frame, fps, game.turn, is_pinch, game.winner)

        # dicas
        tips = "Gestos: Junte polegar+indicador para clicar | 'r' reinicia | 'q' sai"
        cv2.putText(frame, tips, (20, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

        cv2.imshow('Damas por Gestos', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            game.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
