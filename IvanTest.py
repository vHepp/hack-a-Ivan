import cv2 as cv
import numpy as np
from random import randrange as rand
import pygame
import sys

# 0 for webcam feed ; add "path to file"
# for detection in video file
capture = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_eye.xml")


def printMove(command, lastMove, lastMoveCount):
    if command == "l":
        print("MOVE LEFT")
    elif command == "r":
        print("MOVE RIGHT")
    elif command == "u":
        print("QUICK DROP")
    elif command == "d":
        print("MOVE DOWN")
    elif command == "b":
        print("ROTATE")
    elif command == "none":
        print("idle...")

    if command == lastMove:
        print("SAME MOVE")
        lastMoveCount += 1
    else:
        print("different MOVE")
        lastMove = command
        lastMoveCount = 1

    print(lastMove, ':', lastMoveCount)
    return lastMove, lastMoveCount


# TETRIS PART
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Very simple tetris implementation
#
# Control keys:
# Down - Drop stone faster
# Left/Right - Move stone
# Up - Rotate Stone clockwise
# Escape - Quit game
# P - Pause game
#
# Have fun!

# Copyright (c) 2010 "Kevin Chabowski"<kevin@kch42.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# The configuration
config = {
    'cell_size':  20,
    'cols':		  16,
    'rows':		  20,
    'delay':	  600,
    'maxfps':	  15
}

colors = [
    (0,   0,   0),
    (255, 0,   0),
    (0,   150, 0),
    (0,   0,   255),
    (255, 120, 0),
    (255, 255, 0),
    (180, 0,   255),
    (0,   220, 220)
]

# Define the shapes of the single parts
tetris_shapes = [
    [[1, 1, 1],
        [0, 1, 0]],

    [[0, 2, 2],
        [2, 2, 0]],

    [[3, 3, 0],
        [0, 3, 3]],

    [[4, 0, 0],
        [4, 4, 4]],

    [[0, 0, 5],
        [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
        [7, 7]]
]


def rotate_clockwise(shape):
    return [[shape[y][x]
             for y in range(len(shape))]
            for x in range(len(shape[0]) - 1, -1, -1)]


def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False


def remove_row(board, row):
    del board[row]
    return [[0 for i in range(config['cols'])]] + board


def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1][cx+off_x] += val
    return mat1


def new_board():
    board = [[0 for x in range(config['cols'])]
             for y in range(config['rows'])]
    board += [[1 for x in range(config['cols'])]]
    return board


class TetrisApp(object):
    def __init__(self):

        pygame.init()

        pygame.key.set_repeat(250, 25)
        self.width = config['cell_size']*config['cols']
        self.height = config['cell_size']*config['rows']

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # We do not need
        # mouse movement
        # events, so we
        # block them.
        self.init_game()

    def new_stone(self):
        self.stone = tetris_shapes[rand(len(tetris_shapes))]
        self.stone_x = int(config['cols'] / 2 - len(self.stone[0])/2)
        self.stone_y = 0

        if check_collision(self.board,
                           self.stone,
                           (self.stone_x, self.stone_y)):
            self.gameover = True

    def init_game(self):
        self.board = new_board()
        self.new_stone()

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = pygame.font.Font(
                pygame.font.get_default_font(), 12).render(
                    line, False, (255, 255, 255), (0, 0, 0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
                self.width // 2-msgim_center_x,
                self.height // 2-msgim_center_y+i*22))

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x+x) *
                            config['cell_size'],
                            (off_y+y) *
                            config['cell_size'],
                            config['cell_size'],
                            config['cell_size']), 0)

    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > config['cols'] - len(self.stone[0]):
                new_x = config['cols'] - len(self.stone[0])
            if not check_collision(self.board,
                                   self.stone,
                                   (new_x, self.stone_y)):
                self.stone_x = new_x

    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self):
        if not self.gameover and not self.paused:
            self.stone_y += 1
            if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                    self.board,
                    self.stone,
                    (self.stone_x, self.stone_y))
                self.new_stone()
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(
                                self.board, i)
                            break
                    else:
                        break

    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def run(self):
        lastMove = ""
        lastMoveCount = 0
        key_actions = {
            'ESCAPE':	self.quit,
            'LEFT': lambda: self.move(-1),
            'RIGHT': lambda: self.move(+1),
            'DOWN':		self.drop,
            'UP':		self.rotate_stone,
            'p':		self.toggle_pause,
            'SPACE':	self.start_game,
            '': lambda: self.move(0)
        }

        self.gameover = False
        self.paused = False

        pygame.time.set_timer(pygame.USEREVENT+1, config['delay'])
        dont_burn_my_cpu = pygame.time.Clock()
        dont_burn_my_cpu.tick(config['maxfps'])

        while True:

            self.screen.fill((0, 0, 0))

            if self.gameover:
                self.center_msg("""Game Over!Press space to continue""")

            else:
                if self.paused:
                    self.center_msg("Paused")
                else:
                    self.draw_matrix(self.board, (0, 0))
                    self.draw_matrix(self.stone,
                                     (self.stone_x,
                                         self.stone_y))

            ########## FACE RECOGNITION PART ############
            ########## FACE RECOGNITION PART ############
            ########## FACE RECOGNITION PART ############

            ret, frame = capture.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            x, y, w, h = 0, 0, 0, 0
            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.circle(frame, (x + int(w * 0.5), y +
                          int(h * 0.5)), 4, (0, 255, 0), -1)

            eyes = eye_cascade.detectMultiScale(
                gray[y: int(y + h / 1.4), x: (x + w)], 1.1, 4, 0, [30, 30])

            index = 0
            eye_1 = [None, None, None, None]
            eye_2 = [None, None, None, None]

            for ex, ey, ew, eh in eyes:
                if index == 0:
                    eye_1 = [ex, ey, ew, eh]
                elif index == 1:
                    eye_2 = [ex, ey, ew, eh]
                cv.rectangle(
                    frame[y: (y + h), x: (x + w)],
                    (ex, ey),
                    (ex + ew, ey + eh),
                    (0, 0, 255),
                    2,
                )

                index = index + 1

            if (eye_1[0] is not None) and (eye_2[0] is not None):

                if eye_1[0] < eye_2[0]:
                    left_eye = eye_1
                    right_eye = eye_2
                else:
                    left_eye = eye_2
                    right_eye = eye_1

                left_eye_center = (
                    int(left_eye[0] + (left_eye[2] / 2)),
                    int(left_eye[1] + (left_eye[3] / 2)),
                )

                right_eye_center = (
                    int(right_eye[0] + (right_eye[2] / 2)),
                    int(right_eye[1] + (right_eye[3] / 2)),
                )

                left_eye_x = left_eye_center[0]
                left_eye_y = left_eye_center[1]
                right_eye_x = right_eye_center[0]
                right_eye_y = right_eye_center[1]

                delta_x = right_eye_x - left_eye_x
                delta_y = right_eye_y - left_eye_y

                if delta_x == 0:
                    delta_x = 1
                    print("##################DIVISION BY 0####################")

                # Slope of line formula
                angle = np.arctan(delta_y / delta_x)

                # Converting radians to degrees
                angle = (angle * 180) / np.pi

                # Provided a margin of error of 10 degrees
                # (i.e, if the face tilts more than 10 degrees
                # on either side the program will classify as right or left tilt)

                if angle > 10:
                    lastMove, lastMoveCount = printMove(
                        "l", lastMove, lastMoveCount)
                    lastMove = "LEFT"
                    cv.putText(
                        frame,
                        "LEFT TILT :" + str(int(angle)) + " degrees",
                        (20, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                        cv.LINE_4,
                    )

                elif angle < -10:
                    lastMove, lastMoveCount = printMove(
                        "r", lastMove, lastMoveCount)
                    lastMove = "RIGHT"
                    cv.putText(
                        frame,
                        "RIGHT TILT :" + str(int(angle)) + " degrees",
                        (20, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                        cv.LINE_4,
                    )

                else:
                    lastMove, lastMoveCount = printMove(
                        "none", lastMove, lastMoveCount)
                    lastMove = ""
                    cv.putText(
                        frame,
                        "STRAIGHT :",
                        (20, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                        cv.LINE_4,
                    )
            key_actions[lastMove]()
            if cv.waitKey(1) & 0xFF == 27:
                break

            cv.imshow("Frame", frame)
            pygame.display.update()

            # end while loop and run()


if __name__ == '__main__':

    App = TetrisApp()
    App.run()


capture.release()
cv.destroyAllWindows()
