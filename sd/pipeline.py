import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# NOTE: Stable Diffusion can only generate images of dimension (512, 512)
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt: str,
    uncond_prompt:str=None, # negative prompt or empty string if no negative prompt
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    
    '''
    NOTE:
    prompt: str, prompt to tell SD to make generation close to
    uncond_prompt: str, prompt to tell SD to make generation stay away from (usually empty string in classifier-free guidance)
    input_image: img, in case doing image to image
    strength: float, how much attention to pay to starting image if doing image to image
    do_cfg: bool, boolean to do classifier free guidance
    cfg_scale: float, weight of how much model to pay attention to prompt. Goes from 1 to 14
    sampler_name: str, sampler name
    n_inference_steps: int, number of inference steps
    models: pre-trained models
    seed: int, how to initialise random number generator
    device: device to create tensors on
    idle_device: load some model on CUDA, don't need it anymore and can move it to CPU
    tokenizer: tokenizer to tokenize text prompt
    '''

    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        # STEP: Settle text prompt
        if do_cfg:
            # cond_tokens (Prompt to go towards to)
            # STEP: Convert prompt into a list of length Seq_Len=77 with each element being the ID of that token
            # .batch_encode_plus(): To encode the prompt and append padding up to maximum length of 77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # STEP: Convert the prompt from a list of IDs of the tokens to a tensor of the token embeddings
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            
            # STEP: Repeat for the uncond_tokens (Prompt to stay away from)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)

            # STEP: Concatenating along the first, 0th, axis --> in torch.cat, dim=0 by default
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip) # NOTE: To offload to CPU after done with model

        # STEP: Settle the sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        # STEP: Settle image prompt that is passed through the U-Net
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            # scaled between -1 and 1 (NOT betweeen 0 and 1) as per U-Net requirements
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1)) 
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # STEP: Determine how much noise to add to image to decide how much generated image can vary from original image
            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength) # NOTE: creates a time step scheduler, decides an initial noise level to start with
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # NOTE: No image prompt given, start with completely random noise N(0,1). This pertains to text to image generation.
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # STEP: Initialise the sampler
        # NOTE: n_inference_steps determines at which timesteps is being sampled from
        # e.g., if T which is the total number of steps noise is being added,
        # n_inference_steps = 50 means that nosified image is being sampled at 
        # time steps 1000, 980, 960, ... , 40, 20, 0
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # NOTE: Repeat latent twice if doing cfg; one for conditioned latent and one for unconditioned latent
                # .repeat(<a tuple of how many times to repeat along each dimension>)
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2) # NOTE: by default chunk along the 0th dimension
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # STEP: Remove the predicted noise predicted by the U-Net from the nosified image
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # STEP: Decode the latent to get back to original image representation space
        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # NOTE: Same as transformer's formula
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    # NOTE: [:, None] adds a new dimension to the timestamp tensor to go from shape (1,) to (1,1)
    # NOTE: freqs[None] adds a new axis to freqs, converting it from shape (160,) to (160,1)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
