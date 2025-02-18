import pygame
import random

# 初始化 Pygame
pygame.init()

# 屏幕尺寸
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 40
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE  # 定义 ROWS 和 COLS

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 初始化屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("迷宫游戏")

# 迷宫生成算法 (递归回溯)
def generate_maze(rows, cols):
    maze = [[{'top': True, 'right': True, 'bottom': True, 'left': True} for _ in range(cols)] for _ in range(rows)]
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def carve(x, y):
        visited[y][x] = True
        directions = [(0, -1, 'top', 'bottom'), (1, 0, 'right', 'left'), (0, 1, 'bottom', 'top'), (-1, 0, 'left', 'right')]
        random.shuffle(directions)

        for dx, dy, wall, opposite in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny][nx]:
                maze[y][x][wall] = False
                maze[ny][nx][opposite] = False
                carve(nx, ny)

    carve(0, 0)
    return maze

# 渲染迷宫
def draw_maze(maze):
    for y in range(ROWS):
        for x in range(COLS):
            if maze[y][x]['top']:
                pygame.draw.line(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, y * CELL_SIZE), 2)
            if maze[y][x]['right']:
                pygame.draw.line(screen, BLACK, ((x + 1) * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), 2)
            if maze[y][x]['bottom']:
                pygame.draw.line(screen, BLACK, (x * CELL_SIZE, (y + 1) * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), 2)
            if maze[y][x]['left']:
                pygame.draw.line(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE), (x * CELL_SIZE, (y + 1) * CELL_SIZE), 2)

# 玩家控制
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy, maze):
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS:
            if dx == 1 and not maze[self.y][self.x]['right']:
                self.x = nx
            elif dx == -1 and not maze[self.y][self.x]['left']:
                self.x = nx
            elif dy == 1 and not maze[self.y][self.x]['bottom']:
                self.y = ny
            elif dy == -1 and not maze[self.y][self.x]['top']:
                self.y = ny

    def draw(self):
        pygame.draw.rect(screen, GREEN, (self.x * CELL_SIZE + 5, self.y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))

# 游戏结束检测
def check_win(player, end_x, end_y):
    return player.x == end_x and player.y == end_y

# 主游戏循环
def main():
    maze = generate_maze(ROWS, COLS)
    player = Player(0, 0)
    end_x, end_y = COLS - 1, ROWS - 1
    running = True

    while running:
        screen.fill(WHITE)
        draw_maze(maze)
        player.draw()
        pygame.draw.rect(screen, RED, (end_x * CELL_SIZE + 5, end_y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.move(0, -1, maze)
                if event.key == pygame.K_DOWN:
                    player.move(0, 1, maze)
                if event.key == pygame.K_LEFT:
                    player.move(-1, 0, maze)
                if event.key == pygame.K_RIGHT:
                    player.move(1, 0, maze)

        if check_win(player, end_x, end_y):
            print("You Win!")
            running = False

        pygame.display.flip()

    pygame.quit()

import heapq

# A* 算法实现
def a_star(maze, start, end):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# 在游戏循环中添加路径提示
def main():
    maze = generate_maze(ROWS, COLS)
    player = Player(0, 0)
    end_x, end_y = COLS - 1, ROWS - 1
    running = True
    show_path = False

    while running:
        screen.fill(WHITE)
        draw_maze(maze)
        player.draw()
        pygame.draw.rect(screen, RED, (end_x * CELL_SIZE + 5, end_y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))

        if show_path:
            path = a_star(maze, (player.x, player.y), (end_x, end_y))
            if path:
                for x, y in path:
                    pygame.draw.rect(screen, BLUE, (x * CELL_SIZE + 10, y * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.move(0, -1, maze)
                if event.key == pygame.K_DOWN:
                    player.move(0, 1, maze)
                if event.key == pygame.K_LEFT:
                    player.move(-1, 0, maze)
                if event.key == pygame.K_RIGHT:
                    player.move(1, 0, maze)
                if event.key == pygame.K_SPACE:
                    show_path = not show_path

        if check_win(player, end_x, end_y):
            print("You Win!")
            running = False

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

    import numpy as np 
# Q-learning 算法实现
class QLearningAgent:
    def __init__(self, maze, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((ROWS, COLS, 4))  # 上下左右

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state[1], state[0]])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state[1], state[0], action]
        max_future_q = np.max(self.q_table[next_state[1], next_state[0]])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[state[1], state[0], action] = new_q

# 训练智能体
def train_agent(agent, episodes=1000):
    for episode in range(episodes):
        state = (0, 0)
        while state != (COLS - 1, ROWS - 1):
            action = agent.get_action(state)
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            next_state = (state[0] + dx, state[1] + dy)
            if 0 <= next_state[0] < COLS and 0 <= next_state[1] < ROWS:
                reward = -1 if next_state != (COLS - 1, ROWS - 1) else 100
                agent.update_q_table(state, action, reward, next_state)
                state = next_state
            else:
                reward = -10
                agent.update_q_table(state, action, reward, state)

# 在游戏循环中集成智能体
def main():
    maze = generate_maze(ROWS, COLS)
    player = Player(0, 0)
    end_x, end_y = COLS - 1, ROWS - 1
    running = True
    show_path = False
    agent = QLearningAgent(maze)
    train_agent(agent)

    while running:
        screen.fill(WHITE)
        draw_maze(maze)
        player.draw()
        pygame.draw.rect(screen, RED, (end_x * CELL_SIZE + 5, end_y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))

        if show_path:
            path = a_star(maze, (player.x, player.y), (end_x, end_y))
            if path:
                for x, y in path:
                    pygame.draw.rect(screen, BLUE, (x * CELL_SIZE + 10, y * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.move(0, -1, maze)
                if event.key == pygame.K_DOWN:
                    player.move(0, 1, maze)
                if event.key == pygame.K_LEFT:
                    player.move(-1, 0, maze)
                if event.key == pygame.K_RIGHT:
                    player.move(1, 0, maze)
                if event.key == pygame.K_SPACE:
                    show_path = not show_path
                if event.key == pygame.K_a:  # 自动模式
                    state = (player.x, player.y)
                    while state != (end_x, end_y):
                        action = agent.get_action(state)
                        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
                        player.move(dx, dy, maze)
                        state = (player.x, player.y)
                        screen.fill(WHITE)
                        draw_maze(maze)
                        player.draw()
                        pygame.draw.rect(screen, RED, (end_x * CELL_SIZE + 5, end_y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))
                        pygame.display.flip()
                        pygame.time.delay(200)

        if check_win(player, end_x, end_y):
            print("You Win!")
            running = False

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()