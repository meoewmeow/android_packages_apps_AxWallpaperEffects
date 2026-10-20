#version 300 es
precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uTextureColor;
uniform sampler2D uTextureDepth;
uniform vec2 uOffset;
uniform float uParallaxStrength;
uniform float uVignetteStrength;
// Spatial (Rime lock screen): 1 keeps the near subject still and moves the background, as iOS
// Spatial Scenes do, so the clock's depth cut-out stays aligned. 0 is the original Cinematic look.
uniform float uAnchor;
// Spatial with layers: the background with the subject filled in moves, the subject stays.
uniform float uSpatial;
uniform sampler2D uTextureBg;
uniform sampler2D uTextureAlpha;
const float SPATIAL_BG_ZOOM = 1.1;
const float SPATIAL_SUBJECT_GAIN = 0.5;

// 1.12 for Cinematic; 1.0 shows the photo exactly where the clock's depth cut-out expects it.
uniform float uZoom;

float smoothDepth(vec2 uv) {
    vec2 ts = 5.0 / vec2(textureSize(uTextureDepth, 0));
    float d  = texture(uTextureDepth, uv).r;
    d += texture(uTextureDepth, clamp(uv + vec2(ts.x, 0.0), 0.0, 1.0)).r;
    d += texture(uTextureDepth, clamp(uv - vec2(ts.x, 0.0), 0.0, 1.0)).r;
    d += texture(uTextureDepth, clamp(uv + vec2(0.0, ts.y), 0.0, 1.0)).r;
    d += texture(uTextureDepth, clamp(uv - vec2(0.0, ts.y), 0.0, 1.0)).r;
    d += texture(uTextureDepth, clamp(uv + vec2(ts.x, ts.y), 0.0, 1.0)).r;
    d += texture(uTextureDepth, clamp(uv - vec2(ts.x, ts.y), 0.0, 1.0)).r;
    d += texture(uTextureDepth, clamp(uv + vec2(ts.x, -ts.y), 0.0, 1.0)).r;
    d += texture(uTextureDepth, clamp(uv - vec2(ts.x, -ts.y), 0.0, 1.0)).r;
    return d / 9.0;
}

void main() {
    float depth = smoothDepth(vTexCoord);
    if (uSpatial > 0.5) {
        // Farther parts of the background travel further, which keeps it reading as 3D.
        float far = 1.0 - smoothstep(0.05, 0.85, depth);
        vec2 bgUV = (vTexCoord - 0.5) / SPATIAL_BG_ZOOM + 0.5 + uOffset * uParallaxStrength * mix(0.55, 1.0, far);
        vec3 bg = texture(uTextureBg, clamp(bgUV, 0.0, 1.0)).rgb;
        // The subject moves the other way, by SPATIAL_SUBJECT_GAIN of the tilt; the clock between
        // them stays put. SystemUI shifts the clock's cut-out by exactly this much
        // (RimeLockscreenController.SPATIAL_SUBJECT_GAIN): keep the two equal.
        vec2 subjectUV = vTexCoord - uOffset * uParallaxStrength * SPATIAL_SUBJECT_GAIN;
        float a = texture(uTextureAlpha, clamp(subjectUV, 0.0, 1.0)).r;
        fragColor = vec4(mix(bg, texture(uTextureColor, clamp(subjectUV, 0.0, 1.0)).rgb, a), 1.0);
        return;
    }

    float softDepth = smoothstep(0.05, 0.85, depth);

    float vertFade = mix(0.45, 1.0, vTexCoord.y);
    float perspCorrect = mix(1.0, vertFade, softDepth * softDepth);

    float travel = mix(softDepth, 1.0 - softDepth, uAnchor);
    vec2 displacement = uOffset * travel * perspCorrect * uParallaxStrength;

    vec2 baseUV = (vTexCoord - 0.5) / uZoom + 0.5;
    vec2 uv = clamp(baseUV + displacement, 0.0, 1.0);

    vec3 color = texture(uTextureColor, uv).rgb;

    float depthDim = mix(0.82 + 0.18 * depth, 1.0, uAnchor);
    color *= depthDim;

    vec2 vig = vTexCoord - 0.5;
    float vigFactor = 1.0 - dot(vig, vig) * uVignetteStrength * (1.0 - uAnchor);
    color *= clamp(vigFactor, 0.0, 1.0);

    fragColor = vec4(color, 1.0);
}
