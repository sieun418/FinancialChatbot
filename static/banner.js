import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let characterRef = null;

const canvas = document.getElementById("experience-canvas");
const scene = new THREE.Scene();
scene.background = new THREE.Color("#E5E1D3");

const sizes = { width: window.innerWidth, height: window.innerHeight };

// ----- 렌더러 -----
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;

// ----- 카메라 -----
const aspect = sizes.width / sizes.height;
const camera = new THREE.OrthographicCamera(
    -aspect * 1, aspect * 1,
    1, -1,
    0.1, 50
);
camera.position.set(-1.3, 1.0, 1.4);
camera.lookAt(0, 0, 0);
scene.add(camera);

// ----- OrbitControls -----
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.update();

// ----- 조명 -----
const sun = new THREE.DirectionalLight(0xFFF5E5, 3);
sun.position.set(-5, 13, -2);
sun.castShadow = true;
sun.shadow.mapSize.width = 2048;
sun.shadow.mapSize.height = 2048;
sun.shadow.camera.left = -10;
sun.shadow.camera.right = 10;
sun.shadow.camera.top = 10;
sun.shadow.camera.bottom = -10;
sun.shadow.normalBias = 0.1;
scene.add(sun);

// Optional: 조명 helper (개발용)
//const shadowHelper = new THREE.CameraHelper(sun.shadow.camera);
//scene.add(shadowHelper);

// AmbientLight
const ambient = new THREE.AmbientLight(0x404040, 5);
scene.add(ambient);

// ----- 모델 로딩 -----
const loader = new GLTFLoader();

// desk_scene
loader.load('./assets/desk_scene.glb', (glb) => {
    const desk = glb.scene;

    // Group 내부 Mesh에 cast/receive shadow 적용
    desk.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });

    scene.add(desk);
}, undefined, console.error);

// character
loader.load('./assets/character.glb', (glb) => {
    const character = glb.scene;
    character.scale.set(0.25, 0.25, 0.25);
    character.rotation.y = -Math.PI / 2;
    character.position.set(0, 0.21, 0.05);

    character.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });

    scene.add(character);
    // 참조 저장
    characterRef = character;
}, undefined, console.error);

// ----- 리사이즈 대응 -----
window.addEventListener("resize", () => {
    sizes.width = window.innerWidth;
    sizes.height = window.innerHeight;
    const aspect = sizes.width / sizes.height;

    camera.left = -aspect * 1;
    camera.right = aspect * 1;
    camera.top = 1;
    camera.bottom = -1;
    camera.updateProjectionMatrix();

    renderer.setSize(sizes.width, sizes.height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

// ----- 애니메이션 루프 -----
let clock = new THREE.Clock();

function animate() {
    controls.update();

    const elapsed = clock.getElapsedTime();

    if (characterRef) {
        // 위아래로 부드럽게 이동 (0.05 정도 범위)
        const amplitude = 0.05; // 최대 올라가는 높이
        characterRef.position.y = 0.21 + Math.abs(Math.sin(elapsed * 2)) * amplitude;
        // optional: 살짝 회전도 줘서 활발하게 보이게
        characterRef.rotation.y = -Math.PI/2 + Math.sin(elapsed * 2) * 0.1;
    }

//    // ---- 화면 좌우 살짝 흔들기 ----
//    const camAmplitude = 0.2; // 좌우 흔들 폭
//    const camSpeed = 1; // 흔들 속도
//    camera.position.x = -1.3 + Math.sin(elapsed * camSpeed) * camAmplitude;
//    camera.lookAt(0, 0, 0);


    renderer.render(scene, camera);
}
renderer.setAnimationLoop(animate);

const nextBtn = document.getElementById("go-next");
if (nextBtn) {
    nextBtn.addEventListener("click", () => {
        window.location.href = "/page1";  // FastAPI 라우터 주소
    });
}