import torch
from PIL import Image
from torchvision import transforms

def generate_caption(image_path):
    model_path = 'C:/Users/Ansh Varshney/Desktop/Image_Captioning/backend/model/image_captioning_model.pth'
    vocab_path = 'C:/Users/Ansh Varshney/Desktop/Image_Captioning/backend/model/vocab.pkl'

    # Load the model
    checkpoint = torch.load(model_path)
    encoder_state = checkpoint['encoder']
    decoder_state = checkpoint['decoder']
    vocab = checkpoint['vocab']
    
    # Load encoder, decoder, and vocab
    encoder = EncoderCNN()
    encoder.load_state_dict(encoder_state)
    decoder = DecoderRNN(len(vocab))
    decoder.load_state_dict(decoder_state)
    
    encoder.eval()
    decoder.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Generate caption
    features = encoder(image_tensor)
    caption = decoder.sample(features)
    return ' '.join([vocab.idx2word[word] for word in caption])
