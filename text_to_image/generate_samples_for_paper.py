import torch
import json
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import os
from datetime import datetime
import argparse


MODELS = {
        "sd_v1_4": {
            "id": "CompVis/stable-diffusion-v1-4",
            "type": "standard",
            "size": 512
        },
        "sd_v1_5": {
            "id": "runwayml/stable-diffusion-v1-5",
            "type": "standard",
            "size": 512
        },
        "sd_3": {
            "id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "type": "sd3",
            "size": 1024
        },
        "sdxl": {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "type": "xl",
            "size": 1024
        }
    }

def get_pipeline(model_name: str, device):
    if "xl" in model_name.lower():
        return StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
    else:
        return StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)

def generate_images(pipeline, prompt_data, model_name):
    base_dir = f"output_daniel/{datetime.now().strftime('%Y%m%d-%H%M%S')}/Baseline/{model_name}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    image_path = os.path.join(base_dir, "images")
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    score_path = os.path.join(base_dir, "scores.jsonl")

    print(f"Generating images using {model_name}")
    with open(score_path, "w") as score_f:
        for prompt_idx, item in enumerate(prompt_data):
            print(f"{prompt_idx + 1}/{len(prompt_data)}")
            prompt_text = item["prompt"]
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)

            out = pipeline(
                prompt=prompt_text,
                num_inference_steps=50,
                guidance_scale=7.5
            )
            image = out.images[0]

            image_fpath = os.path.join(image_path, f"{prompt_idx}.png")
            image.save(image_fpath)

            result_record ={
                "prompt": prompt_text,
                "prompt_index": prompt_idx,
                "image_path": image_fpath
            }

            score_f.write(json.dumps(result_record) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str)
    args = parser.parse_args()

    prompt_data = [
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, soft daylight, calm blue water surface, slim body shape, smooth skin detail, simple background"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, warm sunset light, reflective orange water, athletic body shape, crisp skin texture, minimal background"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, cool morning light, slightly rippled pale-blue water, slender body shape, natural skin details"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, neutral indoor lighting, deep dark-blue water surface, lean muscular body shape, subtle reflections"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, bright overhead light, clear turquoise water, medium body build, defined skin texture, clean backdrop"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, cloudy diffuse light, muted gray-blue water, soft body outline, gentle skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, strong rim light from behind, shimmering light-touched water, toned body shape, sharp highlights on skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, cool bluish night lighting, still dark water, slim body proportions, light reflections on skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, golden-hour side lighting, textured water surface with soft reflections, athletic lean body shape, warm skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, soft studio lighting, clean simplified water layer, compact body shape, smooth skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor competition pool, bright overhead arena lights, clear lane ropes, dynamic water splashes, athletic body in mid-stroke, crisp skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, outdoor Olympic pool, midday harsh sunlight, strong shadows on water, broad-shouldered swimmer, sharp reflections on wet skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater camera angle at mid-depth, cool blue ambient light, rising bubbles trail, streamlined body position, clear skin and suit texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, early morning fog above open-water lake, soft diffused light, gentle ripples around arms, lean swimmer silhouette, subtle skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, night pool with strong artificial spotlights, dark surroundings, high contrast highlights on water, toned torso, visible droplets frozen in motion"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, rooftop pool in a city skyline, warm evening light, neon reflections in water, medium build swimmer, smooth skin rendering"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, training pool with lane markers and backstroke flags, neutral bright lighting, mild motion blur on arms, defined shoulder muscles, clear water texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open ocean setting, overcast sky, mild waves against the body, endurance swimmer posture, natural skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, crystal-clear tropical lagoon, sunlight beams through water, light caustics on body, slim build, detailed skin pores and droplets"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater low-angle view, strong surface reflections overhead, air bubbles around face, compact body form, smooth skin gradient"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool with greenish industrial lighting, slightly murky water, subtle noise and grain, solid medium build body, realistic wet skin sheen"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, polar training pool with snowy landscape outside windows, cool bluish daylight, light steam above water, lean swimmer, soft skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, bright sunny day at outdoor pool, sharp reflections on lane tiles, water spray trailing behind, muscular arms and back, high clarity skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater close-up at hip level, blurred distant background, fine air bubbles, streamlined legs, subtle skin color variations"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water race start, multiple faint silhouettes in background, cloudy soft light, choppy water surface, powerful arm extension, realistic skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, pool illuminated only by underwater lights, glowing cyan water, dark surroundings, slim swimmer, luminous highlights on wet skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool with large glass windows, strong diagonal sun rays, visible reflections on tiles, medium athletic build, detailed skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, golden-hour outdoor lake, warm orange tones, gentle splash pattern, relaxed training pace posture, natural skin highlights"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, stormy open sea, dramatic dark clouds, rough waves partially obscuring legs, determined body posture, gritty skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, neutral white studio-style lighting over artificial water tank, perfectly calm water surface, simplified background, clean skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater side camera with lane rope visible above, cool teal color palette, bubbles trail behind hand, lean torso, crisp skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, night race with illuminated scoreboard glow, reflections in water, dynamic spray around arm entry, strong back muscles, realistic wet skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, shallow training pool with visible floor tiles, soft overhead light, minimal splash, compact body, subtle muscle definition"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water bay near distant city skyline, hazy sunset, calm rolling waves, long reach stroke, soft skin gradients"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater view with strong god-rays from surface, floating particles in water, streamlined kick, toned legs, soft but detailed skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, frozen motion mid-breath, water droplets suspended around face, neutral indoor lighting, athletic torso, clearly defined skin highlights"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, training session with bright colored lane ropes, bright even pool lighting, subtle motion blur on hand, medium build body, smooth skin rendering"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, overcast afternoon at quiet outdoor pool, slightly desaturated color palette, gentle ripples, relaxed stroke, minimal skin contrast"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, nighttime rooftop pool under city neon signs, mixed cool and warm light reflections, slim silhouette, glossy skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater camera aligned with lane tiles, clear perspective lines, bubbles from exhale, broad shoulders, precise skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water river with greenish water, trees blurred in distant background, soft natural light, medium build swimmer, muted skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, bright midday sun with lens flare, strong contrast shadows on body, sparkling water surface, athletic figure, sharp skin details"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool lit by skylights, soft patterned reflections on water, elongated stroke, relaxed hand entry, smooth skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater slow-motion effect, streaks of bubbles trailing, cool monochrome blue palette, streamlined body, subtle skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open ocean with distant islands, golden late afternoon light, gentle chop around hips, steady breathing position, natural skin rendering"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, competition final with blurred crowd in background, intense overhead lighting, strong water spray, powerful kick, detailed skin reflections"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, early morning indoor practice, slight haze in air, soft overhead lighting, controlled stroke, lightly toned muscles, fine skin gradients"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater view captured from mid-pool, reflections of lane rope on body, thin bubble layer on arms, compact frame, crisp skin texturing"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water swim near rocky coastline, cloudy cool light, scattered white foam on surface, endurance-style posture, neutral skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, pool with colorful flags and timing pads, sharp bright lighting, water droplets streaking backward, toned shoulders, realistic skin highlights"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool with slightly dim lighting, soft shadows on water, smooth arm recovery, medium build, understated skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, sunset-lit lake with orange reflections, calm water, long gliding phase of the stroke, slim silhouette, warm skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater perspective with strong caustic patterns on body, vibrant turquoise water, narrow streamlined position, clearly rendered skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, winter outdoor heated pool, faint steam rising above water, cool blue daylight, relaxed training pace, subtle skin specular highlights"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, night pool illuminated by colored LED lights, multi-colored reflections on skin and water, dynamic kick, athletic frame, glossy wet look"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water channel swim, distant bridge in background, grayish-green water, soft diffused sky light, sturdy build, realistic skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, bright competition start with ripples from nearby lanes, crisp arena lighting, mid-stroke acceleration, lean torso, detailed skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater close-up slightly behind swimmer, focused on back and legs, bubbles trailing feet, cool blue tones, clear skin and muscle detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, calm hotel pool with clean tiled floor, warm indoor lights, gentle water disturbance, compact body line, smooth skin gradients"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, sunrise open-water training, low sun near horizon, golden highlight along body edge, small ripples around arms, soft skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater shot with fish-eye lens distortion, stretching lane lines, air bubbles around head, elongated body, textured skin reflection"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool with glossy white walls, neutral bright lighting, minimal background clutter, steady stroke, medium build, clear skin rendering"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open sea with mild swell, slightly tilted horizon for dynamic feel, muted colors, long distance swimmer posture, realistic skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater view from just below surface, strong reflections of arms, shimmering light patterns, streamlined head position, detailed skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, outdoor pool on cloudy day, soft low contrast lighting, subtle ripples around torso, moderate build, smooth skin transitions"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, competition pool with digital timing board glow, harsh directional lighting, aggressive arm pull, visibly defined back muscles, crisp skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, lake surrounded by dark forest, cool bluish atmosphere, small splashes, relaxed practice pace, gentle skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater with visible lane rope shadow on floor, moderate bubbles, neutral blue-green water, slim build, lightly textured skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, rooftop infinity pool, city lights shimmering on water, soft evening sky gradient, controlled stroke, medium athletic frame, glossy wet skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water triathlon scene, faint silhouettes of other swimmers far behind, flat gray water, efficient stroke form, realistic skin tone"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater shot framed by lane rope and pool wall, strong contrast between body and background, steady flutter kick, crisp skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, afternoon outdoor pool with bright reflections, slight lens flare, spray arcing off fingertips, broad-shouldered swimmer, high-resolution skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, overcast dawn lake, cool desaturated palette, gentle surface disturbances, long glide phase, subtle muscle definition on skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater with high clarity and visible pool drain, thin bubble trail, streamlined profile, toned legs and arms, clean skin highlights"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, night pool lit by a single strong spotlight, high contrast shadows, limited background detail, powerful arm extension, emphasized wet skin shine"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water training in light rain, raindrops dotting surface, soft gray sky, strong forward reach, muted but natural skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, bright training pool with colorful lane dividers, evenly lit environment, moderate splash, balanced body line, clear skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater shot with slight motion blur on feet, dynamic bubble trail, cool aqua colors, slim athletic frame, smooth skin gradients"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, mountain lake with snow-capped peaks in distance, crisp clear air, cold blue water, endurance posture, realistic chilled skin tone"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool during meet warm-up, multiple faint swimmers in back, bright stadium lights, easy relaxed stroke, subtle skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater view near pool floor, body framed against tiled background, rising bubbles, strong leg extension, precise skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, calm harbor water at dusk, reflections of dock lights on surface, slow controlled stroke, medium build, warm-cool mixed skin lighting"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, training pool with visible wall markings, neutral soft lighting, gentle water turbulence, compact streamlined body, even skin tone"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater shot tracking along lane rope, slight distortion from water, shimmering highlights on limbs, lean proportions, clear skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water crossing with distant buoy marker, mild swell, hazy daylight, steady rhythmic stroke, subtle skin specular highlights"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool with blue-tinted lighting, reflections bouncing from walls, vigorous arm pull, muscular build, detailed wet skin rendering"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater almost silhouette against bright surface above, strong caustics around head, minimal bubbles, streamlined torso, soft skin gradients"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, outdoor pool with trees reflecting on water, warm late-afternoon light, light splash trail, balanced body alignment, natural skin tones"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open sea at midday, intense blue everywhere, sun glinting off waves, long-distance pacing posture, subtle skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater shot with camera slightly above body, surface ripples visible, fine bubble patterns, elongated form, high-res skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, indoor pool with reflective ceiling, bright overhead strips of light, mild depth of field blur, athletic frame, smooth skin highlights"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open-water bay under soft pink sunset, calm surface, gentle wake trail, relaxed gliding stroke, warm skin coloration"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater capture focused on arm extension, background fading into deep blue, bubble streak along forearm, lean torso, detailed skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, outdoor lap pool with checkerboard tiles, crisp midday sun, sharp reflections on water, controlled mid-stroke, realistic skin rendering"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, open river swim with bridge shadow on water, muted daylight, shallow ripples around hips, medium build, soft skin gradients"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater image from behind the swimmer, visible trailing legs and feet, bubble ribbons, cool cyan palette, fine skin detail"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, night pool with overhead stadium lights and dark stands, high contrast light on shoulders, energetic kick, toned frame, glossy wet skin"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, coastal open-water scene with distant lighthouse, soft daylight, mixed green-blue water, efficient stroke, natural skin shading"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, underwater capture with dramatic diagonal light beams, suspended particles, tight streamlined position, defined muscles, detailed skin texture"},
        {
            "prompt": "A person doing freestyle swimming stroke, side view, full body, quiet indoor practice with muted colors, gentle ripples only around hands, relaxed stroke rhythm, compact body build, smooth subtle skin detail"}
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = get_pipeline(MODELS[args.model_name]["id"], device)
    generate_images(pipeline, prompt_data, args.model_name)






