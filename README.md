# Adversarially Trained Diffusion Models for Generative Steganography
This project explores the possibility of improving the efficacy of generative steganography through adversarially trained diffusion models in tandem with diffusion models.

## What is steganography?

Steganography refers to embedding data into cover media. For example, consider the following situation: Bob wants to send an image of his dog to Alice. Instead of encrypting the data using some protocol, he can embed his image of a dog into an image of a cat, which Alice is then able to transform back into the dog. This image use-case is what this project explores.

## and, generative steganography?

A traditional diffusion model begins with a latent space, and generates images from there. Generative steganography (GS) refers to projecting messages into that latent space, and effectively generating an image around the embedded data. Pretty cool, right?

## So, what does this do?

Well, GANs typically score higher on data-embedding benchmark tests, and diffusion models usually score higher on image quality benchmark tests, so why not use both? This project explores that possibility.
