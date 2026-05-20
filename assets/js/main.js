    (function () {
    const canvas = document.getElementById('topology-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    let w, h;
    let layers = [];     
    let dataPulses = []; 
    let scanners = [];   
    let symbols = [];    
    let fourierWaves = []; 
    let time = 0;

    // BANCO 1: Exclusivo para os Scanners 2D (Processamento de Imagem)
    const BOX_LABELS = [
        'FFT 2D Spectrum'
    ];

    // BANCO 2: Exclusivo para as Equações e Matrizes Flutuantes de Fundo
    const SYMBOL_POOL = [
        'eⁱᵡ + 1 = 0', 
        'ℱ{f(t)} = ∫ f(t)e⁻ⁱʷᵗdt', 
        'F(u,v) = ∬ f(x,y)e⁻ⁱ²ᵖⁱ⁽ᵘᵡ⁺ᵛʸ⁾dxdy',
        '(f * g)(t) = ∫ f(τ)g(t-τ)dτ',
        'g(x,y) = ω * f(x,y)', 
        '∇²f = ∂²f/∂x² + ∂²f/∂y²',
        'Tensor 𝐗 ∈ ℝᴮ × ᶜ × ᴴ × ᵂ',
        '⎡ 1  0 -1 ⎤\n⎢ 2  0 -2 ⎥\n⎣ 1  0 -1 ⎦', // Sobel
        '⎡ 0 -1  0 ⎤\n⎢-1  4 -1 ⎥\n⎣ 0 -1  0 ⎦', // Laplacian
        '⎡ w₁₁  w₁₂ ⎤\n⎣ w₂₁  w₂₂ ⎦'            // Weights
    ];

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }

    // Camadas da rede convolucional/deep learning
    function initNetwork() {
        layers = [];
        const layerSizes = [4, 7, 7, 5, 3]; 
        const layerCount = layerSizes.length;
        const paddingX = w * 0.18; 
        const layerSpacing = (w - paddingX * 2) / (layerCount - 1);

        for (let i = 0; i < layerCount; i++) {
            const numNodes = layerSizes[i];
            const layerNodes = [];
            const layerX = paddingX + i * layerSpacing;
            const paddingY = h * 0.22;
            const nodeSpacing = (h - paddingY * 2) / (numNodes - 1 || 1);

            for (let j = 0; j < numNodes; j++) {
                const layerY = numNodes === 1 ? h / 2 : paddingY + j * nodeSpacing;
                layerNodes.push({
                    x: layerX,
                    y: layerY,
                    baseY: layerY,
                    r: i === 0 || i === layerCount - 1 ? 4 : 2.5,
                    id: `${i}_${j}`,
                    layerIndex: i
                });
            }
            layers.push(layerNodes);
        }
    }

    // Ondas senoidais no rodapé (Simulando decomposição harmônica)
    function initFourierWaves() {
        fourierWaves = [];
        for (let i = 0; i < 3; i++) {
            fourierWaves.push({
                amplitude: 12 + i * 8,
                frequency: 0.008 * (i + 1),
                phase: i * Math.PI / 3,
                yCenter: h * 0.85 + (i * 12), 
                color: `rgba(79, 140, 255, ${0.05 - (i * 0.01)})`
            });
        }
    }

    // Inicialização dos blocos de processamento (Scanners de Imagem)
    function initScanners() {
        scanners = [];
        // const count = Math.max(2, Math.floor(w / 500));
        const count = 1
        for (let i = 0; i < count; i++) {
            scanners.push({
                x: Math.random() * (w - 180),
                y: Math.random() * (h - 180),
                sizeW: 120 + Math.random() * 50, 
                sizeH: 120 + Math.random() * 50,
                vx: (Math.random() - 0.5) * 0.25,
                vy: (Math.random() - 0.5) * 0.25,
                label: BOX_LABELS[Math.floor(Math.random() * BOX_LABELS.length)], 
                alpha: 0.08 + Math.random() * 0.12
            });
        }
    }

    // Inicialização das Equações Flutuantes Ambientais
    function initSymbols() {
        symbols = [];
        const count = Math.max(8, Math.floor(w / 160)); 
        for (let i = 0; i < count; i++) {
            symbols.push({
                text: SYMBOL_POOL[Math.floor(Math.random() * SYMBOL_POOL.length)],
                x: Math.random() * w,
                y: Math.random() * h,
                vx: (Math.random() - 0.5) * 0.05, 
                vy: -0.06 - Math.random() * 0.02,  // Subida lenta contínua
                alpha: 0.04 + Math.random() * 0.06, 
                size: 11 + Math.random() * 3,
            });
        }
    }

    function spawnPulse() {
        if (layers.length < 2) return;
        const startNode = layers[0][Math.floor(Math.random() * layers[0].length)];
        const targetNode = layers[1][Math.floor(Math.random() * layers[1].length)];
        
        dataPulses.push({
            startX: startNode.x,
            startY: startNode.y,
            currentX: startNode.x,
            currentY: startNode.y,
            targetNode: targetNode,
            speed: 0.012 + Math.random() * 0.008,
            progress: 0,
            currentLayer: 0
        });
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);
        time += 0.004;

        // 1. Desenha Ondas de Fourier no rodapé
        for (const wave of fourierWaves) {
            ctx.beginPath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = wave.color;
            for (let x = 0; x < w; x += 5) {
                const y = wave.yCenter + Math.sin(x * wave.frequency + wave.phase + time) * wave.amplitude;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // 2. Desenha as Sinapses da Rede
        for (let l = 0; l < layers.length - 1; l++) {
            for (const n1 of layers[l]) {
                for (const n2 of layers[l + 1]) {
                    const alpha = 0.05 + (Math.sin(time + n1.y * 0.02) * 0.02);
                    ctx.beginPath();
                    ctx.moveTo(n1.x, n1.y);
                    ctx.lineTo(n2.x, n2.y);
                    ctx.strokeStyle = `rgba(79, 140, 255, ${alpha})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        // 3. Desenha Pulsos de Tensores
        for (const pulse of dataPulses) {
            ctx.beginPath();
            ctx.arc(pulse.currentX, pulse.currentY, 1.8, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 230, 118, 0.8)`; 
            ctx.fill();
        }

        // 4. Desenha Neurónios
        for (let l = 0; l < layers.length; l++) {
            for (const node of layers[l]) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
                if (l === 0) ctx.fillStyle = 'rgba(0, 230, 118, 0.5)';
                else if (l === layers.length - 1) ctx.fillStyle = 'rgba(255, 109, 0, 0.5)'; 
                else ctx.fillStyle = 'rgba(79, 140, 255, 0.35)';
                ctx.fill();
            }
        }

        // 5. DESENHAR SCANNERS (FFT 2D DO PROPRIO PROJETO)
        for (const box of scanners) {
            ctx.strokeStyle = `rgba(0, 230, 118, ${box.alpha})`;
            ctx.lineWidth = 1;
            ctx.strokeRect(box.x, box.y, box.sizeW, box.sizeH);

            ctx.fillStyle = `rgba(0, 230, 118, ${box.alpha + 0.2})`;
            const len = 8;
            ctx.fillRect(box.x, box.y, len, 1.5); ctx.fillRect(box.x, box.y, 1.5, len);
            ctx.fillRect(box.x + box.sizeW - len, box.y, len, 1.5); ctx.fillRect(box.x + box.sizeW, box.y, 1.5, len);
            ctx.fillRect(box.x, box.y + box.sizeH, len, -1.5); ctx.fillRect(box.x, box.y + box.sizeH - len, 1.5, len);
            ctx.fillRect(box.x + box.sizeW - len, box.y + box.sizeH, len, -1.5); ctx.fillRect(box.x + box.sizeW, box.y + box.sizeH - len, 1.5, len);

            ctx.font = `9px 'JetBrains Mono', monospace`;
            ctx.fillStyle = `rgba(0, 230, 118, ${box.alpha + 0.3})`;
            ctx.fillText(box.label, box.x + 4, box.y - 4);

            const centerX = box.x + box.sizeW / 2;
            const centerY = box.y + box.sizeH / 2;
            
            const pulseIntensity = 0.3 + 0.15 * Math.sin(time * 5);
            const gradient = ctx.createRadialGradient(centerX, centerY, 2, centerX, centerY, box.sizeW * 0.45);
            gradient.addColorStop(0, `rgba(0, 230, 118, ${box.alpha * 2.5 * pulseIntensity})`);
            gradient.addColorStop(0.15, `rgba(0, 230, 118, ${box.alpha * 0.8})`);
            gradient.addColorStop(0.4, `rgba(0, 230, 118, ${box.alpha * 0.15})`);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(box.x + 2, box.y + 2, box.sizeW - 4, box.sizeH - 4);

            ctx.strokeStyle = `rgba(0, 230, 118, ${box.alpha * 0.4})`;
            ctx.lineWidth = 0.5;
            
            ctx.beginPath();
            ctx.moveTo(box.x + 10, centerY); ctx.lineTo(box.x + box.sizeW - 10, centerY);
            ctx.moveTo(centerX, box.y + 10); ctx.lineTo(centerX, box.y + box.sizeH - 10);
            ctx.stroke();

            ctx.fillStyle = `rgba(0, 230, 118, ${box.alpha * 0.8})`;
            const peakDistance = (box.sizeW * 0.25);
            const orbitOffset = Math.sin(time * 2) * 4;
            
            ctx.beginPath();
            ctx.arc(centerX + peakDistance, centerY + orbitOffset, 1.5, 0, Math.PI * 2);
            ctx.arc(centerX - peakDistance, centerY - orbitOffset, 1.5, 0, Math.PI * 2);
            ctx.arc(centerX, centerY + peakDistance * 0.8 + orbitOffset, 1.5, 0, Math.PI * 2);
            ctx.arc(centerX, centerY - peakDistance * 0.8 - orbitOffset, 1.5, 0, Math.PI * 2);
            ctx.fill();
        }

// 6. DESENHAR EQUAÇÕES E MATRIZES INDEPENDENTES (CORRIGIDO: MAIS LEGÍVEL)
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (const sym of symbols) {
            // Modificado: O seno agora oscila de forma bem mais lenta (time * 0.8) 
            // e o brilho nunca cai para zero (mínimo de 0.55 de visibilidade)
            const flicker = 0.55 + 0.45 * Math.sin(time * 0.8 + sym.x * 0.05);
            
            ctx.font = `${sym.size}px 'JetBrains Mono', monospace`;
            
            // Modificado: Multiplicamos por 2.0 para dar mais contraste e nitidez ao azul/branco
            ctx.fillStyle = `rgba(255, 255, 255, ${sym.alpha * flicker * 2.2})`;
            
            if (sym.text.includes('\n')) {
                const lines = sym.text.split('\n');
                const lineHeight = sym.size * 1.3; // Espaçamento entre linhas ligeiramente maior
                lines.forEach((line, index) => {
                    ctx.fillText(line, sym.x, sym.y + (index - (lines.length - 1) / 2) * lineHeight);
                });
            } else {
                ctx.fillText(sym.text, sym.x, sym.y);
            }
        }
    }

    function update() {
        for (let l = 0; l < layers.length; l++) {
            for (const node of layers[l]) {
                node.y = node.baseY + Math.sin(time + node.baseY * 0.04) * 6;
            }
        }

        for (let i = dataPulses.length - 1; i >= 0; i--) {
            const p = dataPulses[i];
            p.progress += p.speed;
            p.currentX = p.startX + (p.targetNode.x - p.startX) * p.progress;
            p.currentY = p.startY + (p.targetNode.y - p.startY) * p.progress;

            if (p.progress >= 1) {
                const nextLayerIdx = p.currentLayer + 1;
                if (nextLayerIdx < layers.length - 1) {
                    const nextTarget = layers[nextLayerIdx + 1][Math.floor(Math.random() * layers[nextLayerIdx + 1].length)];
                    p.startX = p.targetNode.x;
                    p.startY = p.targetNode.y;
                    p.targetNode = nextTarget;
                    p.currentLayer = nextLayerIdx;
                    p.progress = 0;
                } else {
                    dataPulses.splice(i, 1);
                }
            }
        }

        if (Math.random() < 0.05 && dataPulses.length < 30) {
            spawnPulse();
        }

        for (const box of scanners) {
            box.x += box.vx;
            box.y += box.vy;
            if (box.x < 0 || box.x + box.sizeW > w) box.vx *= -1;
            if (box.y < 0 || box.y + box.sizeH > h) box.vy *= -1;
        }

        // Movimentação e reciclagem correta dos símbolos
for (const sym of symbols) {
            sym.x += sym.vx;
            sym.y += sym.vy;
            
            // Se sumir no topo (y < -60), reseta no rodapé de forma segura
            if (sym.y < -60) {
                sym.y = h + 60;
                sym.x = Math.random() * w;
                sym.text = SYMBOL_POOL[Math.floor(Math.random() * SYMBOL_POOL.length)];
                
                // --- MATCH THE SLOWER SPEED HERE TOO ---
                sym.vx = (Math.random() - 0.5) * 0.05; 
                sym.vy = -0.02 - Math.random() * 0.04;
                
                sym.alpha = 0.04 + Math.random() * 0.06;
            }
        }
    }

    function animate() {
        update();
        draw();
        requestAnimationFrame(animate);
    }

    resize();
    initNetwork();
    initFourierWaves();
    initScanners();
    initSymbols();
    animate();

    window.addEventListener('resize', () => {
        resize();
        initNetwork();
        initFourierWaves();
        initScanners();
        initSymbols();
    });
})();

// ===== Efeito de Máquina de Escrever Embutido =====
(function () {
    const target = document.getElementById('typing-target');
    if (!target) return;

    const phrases = [
        'PhD Candidate in Electrical Engineering',
        'Specialist in Computer Vision & Deep Learning',
        'Senior Data Engineer'
    ];

    let phraseIdx = 0;
    let charIdx = 0;
    let isDeleting = false;
    let delay = 80;

    function type() {
        const current = phrases[phraseIdx];

        if (isDeleting) {
            target.innerHTML = current.substring(0, charIdx - 1) + '<span class="cursor"></span>';
            charIdx--;
            delay = 30;
        } else {
            target.innerHTML = current.substring(0, charIdx + 1) + '<span class="cursor"></span>';
            charIdx++;
            delay = 70;
        }

        if (!isDeleting && charIdx === current.length) {
            delay = 2000;
            isDeleting = true;
        } else if (isDeleting && charIdx === 0) {
            isDeleting = false;
            phraseIdx = (phraseIdx + 1) % phrases.length;
            delay = 400;
        }

        setTimeout(type, delay);
    }

    type();
})();