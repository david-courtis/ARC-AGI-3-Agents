"""Web-based visualization for local game simulations."""

import time
from dataclasses import dataclass, asdict
from typing import Type
from flask import Flask, render_template_string, jsonify

from agents.structs import FrameData, GameAction, GameState
from local_simulator.core.base_game import BaseGame


@dataclass
class RecordedFrame:
    frame_index: int
    action_name: str
    state: str
    score: int
    level: int
    total_levels: int
    pixels: list[list[list[int]]]
    timestamp: float
    action_counter: int = 0


class GameRecorder:
    def __init__(self):
        self.frames: list[RecordedFrame] = []
        self.start_time = 0.0
    
    def start(self):
        self.frames = []
        self.start_time = time.time()
    
    def record_step(self, frame_index: int, action: GameAction, frame_data: FrameData,
                    level: int, total_levels: int, action_counter: int):
        self.frames.append(RecordedFrame(
            frame_index=frame_index,
            action_name=action.name,
            state=frame_data.state.name,
            score=frame_data.score,
            level=level,
            total_levels=total_levels,
            pixels=frame_data.frame,
            timestamp=time.time() - self.start_time,
            action_counter=action_counter
        ))
    
    def to_json(self) -> list[dict]:
        return [asdict(f) for f in self.frames]


def run_and_record(game: BaseGame, agent_class: Type, max_actions: int = 200) -> GameRecorder:
    import random
    
    recorder = GameRecorder()
    recorder.start()
    
    frame = game.reset()
    recorder.record_step(0, GameAction.RESET, frame, game.current_level_index + 1, game.get_level_count(), 0)
    
    seed = int(time.time() * 1000000) + hash(game.game_id) % 1000000
    random.seed(seed)
    
    frame_counter = 0
    action_counter = 0
    
    while action_counter < max_actions:
        if frame.state == GameState.WIN:
            break
        
        if frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            action = random.choice([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
        
        current_action_num = action_counter + 1
        
        if action == GameAction.RESET:
            frame = game.reset()
            frame_counter += 1
            recorder.record_step(frame_counter, action, frame, game.current_level_index + 1, game.get_level_count(), current_action_num)
        else:
            if hasattr(game, 'get_all_slide_frames'):
                slide_frames = game.get_all_slide_frames(action)
                for sf in slide_frames:
                    frame_counter += 1
                    recorder.record_step(frame_counter, action, sf, game.current_level_index + 1, game.get_level_count(), current_action_num)
                frame = slide_frames[-1] if slide_frames else frame
            else:
                frame = game.step(action)
                frame_counter += 1
                recorder.record_step(frame_counter, action, frame, game.current_level_index + 1, game.get_level_count(), current_action_num)
        
        action_counter += 1
        if frame.state == GameState.WIN:
            break
    
    return recorder


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ARC-AGI-3 Local Simulator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container { max-width: 700px; width: 100%; }
        h1 { text-align: center; margin-bottom: 20px; font-size: 1.5rem; color: #00d9ff; }
        .game-display {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .canvas-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        #gameCanvas {
            border-radius: 8px;
            image-rendering: pixelated;
            box-shadow: 0 4px 20px rgba(0,217,255,0.2);
        }
        .info-panel {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        .info-box {
            background: rgba(255,255,255,0.08);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .info-box .label { font-size: 0.75rem; color: #888; margin-bottom: 4px; }
        .info-box .value { font-size: 1.2rem; font-weight: 600; color: #00d9ff; }
        .info-box.win .value { color: #00ff88; }
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        button {
            background: linear-gradient(135deg, #00d9ff 0%, #00a8cc 100%);
            border: none;
            color: #fff;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,217,255,0.4); }
        button.secondary { background: rgba(255,255,255,0.1); color: #fff; }
        .slider-container { flex: 1; max-width: 400px; display: flex; align-items: center; gap: 12px; }
        #frameSlider {
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
        }
        #frameSlider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d9ff;
            border-radius: 50%;
            cursor: pointer;
        }
        .speed-control { display: flex; align-items: center; gap: 8px; }
        .speed-control label { font-size: 0.85rem; color: #888; }
        #speedSelect {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
        }
        .timeline {
            height: 40px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            position: relative;
            cursor: pointer;
        }
        .timeline-marker {
            position: absolute;
            height: 100%;
            width: 2px;
            background: #00d9ff;
            pointer-events: none;
            z-index: 10;
        }
        .timeline-line {
            position: absolute;
            height: 100%;
            width: 2px;
            background: rgba(0, 255, 136, 0.7);
        }
        .timeline-line.level-change { background: #ffd700; width: 3px; }
        .legend {
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-top: 12px;
            font-size: 0.8rem;
            color: #888;
        }
        .legend span { display: flex; align-items: center; gap: 6px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 2px; }
        .legend-dot.action { background: rgba(0, 255, 136, 0.7); }
        .legend-dot.level { background: #ffd700; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 ARC-AGI-3 Local Simulator</h1>
        <div class="game-display">
            <div class="canvas-container">
                <canvas id="gameCanvas" width="512" height="512"></canvas>
            </div>
            <div class="info-panel">
                <div class="info-box">
                    <div class="label">Action #</div>
                    <div class="value" id="actionNumVal">0</div>
                </div>
                <div class="info-box">
                    <div class="label">Score</div>
                    <div class="value" id="scoreVal">0</div>
                </div>
                <div class="info-box">
                    <div class="label">Level</div>
                    <div class="value" id="levelVal">1/3</div>
                </div>
                <div class="info-box action">
                    <div class="label">Action</div>
                    <div class="value" id="actionVal">RESET</div>
                </div>
            </div>
            <div class="controls">
                <button class="secondary" onclick="stepBack()">⏮ Prev</button>
                <button onclick="togglePlay()" id="playBtn">▶ Play</button>
                <button class="secondary" onclick="stepForward()">Next ⏭</button>
                <div class="slider-container">
                    <input type="range" id="frameSlider" min="0" value="0">
                </div>
                <div class="speed-control">
                    <label>Speed:</label>
                    <select id="speedSelect">
                        <option value="2000">0.5x</option>
                        <option value="1000">1x</option>
                        <option value="500" selected>2x</option>
                        <option value="200">5x</option>
                        <option value="100">10x</option>
                        <option value="50">20x</option>
                    </select>
                </div>
            </div>
            <div class="timeline" id="timeline">
                <div class="timeline-marker" id="timelineMarker"></div>
            </div>
            <div class="legend">
                <span><div class="legend-dot action"></div> Action</span>
                <span><div class="legend-dot level"></div> Level Change</span>
            </div>
        </div>
    </div>
    
    <script>
        let frames = [];
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;
        
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const slider = document.getElementById('frameSlider');
        const actionNumVal = document.getElementById('actionNumVal');
        const scoreVal = document.getElementById('scoreVal');
        const levelVal = document.getElementById('levelVal');
        const actionVal = document.getElementById('actionVal');
        const playBtn = document.getElementById('playBtn');
        const speedSelect = document.getElementById('speedSelect');
        const timeline = document.getElementById('timeline');
        const timelineMarker = document.getElementById('timelineMarker');
        
        async function loadFrames() {
            const response = await fetch('/api/frames');
            frames = await response.json();
            slider.max = frames.length - 1;
            buildTimeline();
            showFrame(0);
        }
        
        function buildTimeline() {
            timeline.querySelectorAll('.timeline-line').forEach(e => e.remove());
            if (frames.length === 0) return;
            
            let prevActionId = null;
            let prevLevel = null;
            
            frames.forEach((frame, i) => {
                const actionId = frame.action_counter;
                const level = frame.level;
                
                // Draw line at action boundaries
                if (prevActionId !== null && actionId !== prevActionId) {
                    const line = document.createElement('div');
                    line.className = 'timeline-line';
                    line.style.left = (i / frames.length * 100) + '%';
                    line.title = `Action ${actionId}`;
                    line.onclick = (e) => { e.stopPropagation(); showFrame(i); };
                    timeline.appendChild(line);
                }
                
                // Draw level change markers
                if (prevLevel !== null && level !== prevLevel) {
                    const line = document.createElement('div');
                    line.className = 'timeline-line level-change';
                    line.style.left = (i / frames.length * 100) + '%';
                    line.title = `Level ${level}`;
                    timeline.appendChild(line);
                }
                
                prevActionId = actionId;
                prevLevel = level;
            });
            
            timeline.onclick = (e) => {
                if (e.target.classList.contains('timeline-line')) return;
                const rect = timeline.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const pct = Math.max(0, Math.min(1, x / rect.width));
                showFrame(Math.floor(pct * (frames.length - 1)));
            };
        }
        
        function showFrame(index) {
            if (index < 0 || index >= frames.length) return;
            currentFrame = index;
            const frame = frames[index];
            
            const pixels = frame.pixels;
            const scale = 512 / 64;
            if (pixels && pixels.length > 0) {
                for (let y = 0; y < 64; y++) {
                    for (let x = 0; x < 64; x++) {
                        const [r, g, b] = pixels[y][x];
                        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                        ctx.fillRect(x * scale, y * scale, scale, scale);
                    }
                }
            }
            
            actionNumVal.textContent = frame.action_counter;
            scoreVal.textContent = frame.score;
            levelVal.textContent = `${frame.level}/${frame.total_levels}`;
            actionVal.textContent = frame.action_name;
            slider.value = index;
            timelineMarker.style.left = (index / frames.length * 100) + '%';
            
            const scoreBox = scoreVal.closest('.info-box');
            scoreBox.classList.toggle('win', frame.state === 'WIN');
        }
        
        function togglePlay() {
            isPlaying = !isPlaying;
            playBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            if (isPlaying) {
                const speed = parseInt(speedSelect.value);
                playInterval = setInterval(() => {
                    if (currentFrame >= frames.length - 1) { togglePlay(); return; }
                    showFrame(currentFrame + 1);
                }, speed);
            } else {
                clearInterval(playInterval);
            }
        }
        
        function stepForward() { showFrame(currentFrame + 1); }
        function stepBack() { showFrame(currentFrame - 1); }
        
        slider.addEventListener('input', (e) => showFrame(parseInt(e.target.value)));
        
        speedSelect.addEventListener('change', () => {
            if (isPlaying) {
                clearInterval(playInterval);
                const speed = parseInt(speedSelect.value);
                playInterval = setInterval(() => {
                    if (currentFrame >= frames.length - 1) { togglePlay(); return; }
                    showFrame(currentFrame + 1);
                }, speed);
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ' || e.key === 'k') { e.preventDefault(); togglePlay(); }
            else if (e.key === 'ArrowRight' || e.key === 'l') { stepForward(); }
            else if (e.key === 'ArrowLeft' || e.key === 'j') { stepBack(); }
        });
        
        loadFrames();
    </script>
</body>
</html>
"""


def create_app(recorder: GameRecorder) -> Flask:
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/frames')
    def get_frames():
        return jsonify(recorder.to_json())
    
    return app


def run_visualization(game: BaseGame, agent_class: Type, max_actions: int = 200, 
                      port: int = 5000, open_browser: bool = True):
    print("Recording gameplay...")
    recorder = run_and_record(game, agent_class, max_actions)
    print(f"Recorded {len(recorder.frames)} frames")
    
    app = create_app(recorder)
    
    if open_browser:
        import webbrowser
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    
    print(f"\n🎮 Open http://localhost:{port} to view the replay")
    print("Press Ctrl+C to stop the server\n")
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == "__main__":
    import argparse
    from agents.templates.random_agent import Random
    
    parser = argparse.ArgumentParser(description='Run local simulator visualization')
    parser.add_argument('--game', type=str, choices=['maze', 'ice'], default='maze')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    if args.game == 'ice':
        from local_simulator.games.ice_sliding import IceSlidingPuzzle
        game = IceSlidingPuzzle()
    else:
        from local_simulator.games.simple_maze import SimpleMaze
        game = SimpleMaze()
    
    run_visualization(game, Random, max_actions=300, port=args.port)
