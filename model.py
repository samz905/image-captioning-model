import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        
        self.A = nn.Linear(attention_dim, 1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))
        
        attention_weights = self.A(combined_states)
        attention_weights = attention_weights.squeeze(2)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context_vector = features * attention_weights.unsqueeze(2)
        context_vector = context_vector.sum(dim=1)
        
        return attention_weights, context_vector

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
    def forward(self, features, captions):
        embeds = self.embedding(captions)
        
        h, c = self.init_hidden_state(features)
        
        seq_length = len(captions[0]) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)
        
        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            output = self.fcn(self.drop(h))
            
            preds[:, s] = output
            alphas[:, s] = alpha
            
        return preds, alphas
        
    def generate_caption(self, features, max_len=20, vocab=None):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
        
        all_predictions = []
        
        word = torch.tensor(vocab.stoi['< SOS >']).view(1, -1).to(features.device)
        embeds = self.embedding(word)
        
        alphas = []
        
        for _ in range(max_len):
            alpha, context = self.attention(features, h)
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            
            output = output.view(batch_size, -1)
            
            predicted = output.argmax(dim=1)
            
            if vocab.itos[predicted.item()] == "<EOS>":
                break
                
            all_predictions.append(predicted.item())
            
            embeds = self.embedding(predicted).unsqueeze(1)
            
        return [vocab.itos[idx] for idx in all_predictions], alphas
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
