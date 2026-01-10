
window.addEventListener('DOMContentLoaded', () => {
  tsParticles.load('tsparticles', {
    background: {
      color: 'transparent',
    },
    fpsLimit: 60,
    interactivity: {
      events: {
        onHover: { enable: true, mode: 'repulse' },
        resize: true
      },
      modes: {
        repulse: { distance: 80, duration: 0.4 }
      }
    },
    particles: {
      color: { value: '#ffb347' },
      links: {
        color: '#ffb347',
        distance: 120,
        enable: true,
        opacity: 0.3,
        width: 1
      },
      collisions: { enable: false },
      move: {
        direction: 'none',
        enable: true,
        outModes: { default: 'bounce' },
        random: false,
        speed: 1.2,
        straight: false
      },
      number: { density: { enable: true, area: 800 }, value: 40 },
      opacity: { value: 0.5 },
      shape: { type: 'circle' },
      size: { value: { min: 2, max: 4 } }
    },
    detectRetina: true
  });
});
