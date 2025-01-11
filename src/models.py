
#Modificado para 3 canales
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(TwoCNN, self).__init__()
        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        #print("Input to TwoCNN:", x.shape)
        out = self.body(x)
        #print("Output from Conv2D in TwoCNN:", out.shape)
        out = torch.add(out, x)  # Residual connection
        #print("Output after Residual Connection in TwoCNN:", out.shape)
        return out


class ThreeCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(ThreeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)

        self.body_spatial = nn.Sequential(
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)),
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x):
        #print("Input to ThreeCNN:", x.shape)
        for i, layer in enumerate(self.body_spatial):
            x = layer(x)
            #print(f"Output from Conv2D layer {i+1} in ThreeCNN:", x.shape)
            if i == 0:
                x = self.act(x)
                #print(f"Output after activation in ThreeCNN (layer {i+1}):", x.shape)
        return x

class SFCSR(nn.Module):
    def __init__(self, args):
        super(SFCSR, self).__init__()
        upscale_factor = int(args.upscale_factor)
        n_feats = args.n_feats
        self.n_module = args.n_module

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.gamma_X = nn.Parameter(torch.ones(self.n_module))
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module))
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))

        # Entrada directa para RGB
        self.head = wn(nn.Conv2d(3, n_feats, kernel_size=3, stride=1, padding=1))

        # Bloques principales
        self.twoCNN = TwoCNN(wn, n_feats)
        self.threeCNN = ThreeCNN(wn, n_feats)

        # Capa de remuestreo para salida final
        self.tail = nn.Sequential(
            wn(nn.Conv2d(n_feats, 3 * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        #print("Input to SFCSR:", x.shape)

        # Head
        x = self.head(x)
        #print("Output from Head:", x.shape)

        # Aplicar TwoCNN varias veces
        for i in range(self.n_module):
            x = self.twoCNN(x)

        # Aplicar ThreeCNN
        x = self.threeCNN(x)

        #print("Output after all modules:", x.shape)

        # Tail (generar superresolución)
        x = self.tail(x)
        #print("Output from Tail:", x.shape)

        return x

    
###########################
           #MCNET
###########################

class BasicConv3d(nn.Module):
    def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0,0,0)):
        super(BasicConv3d, self).__init__()

        self.conv = wn(
            nn.Conv3d(in_channel, out_channel,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding)
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        #print(f"[BasicConv3d] Input: {x.shape}")
        x = self.conv(x)
        #print(f"[BasicConv3d] After Conv3d: {x.shape}")
        x = self.relu(x)
        #print(f"[BasicConv3d] After ReLU: {x.shape}")
        return x

class S3Dblock(nn.Module):
    """
    Bloque 3D con dos convoluciones 3D sucesivas.
    """
    def __init__(self, wn, n_feats):
        super(S3Dblock, self).__init__()
        self.conv = nn.Sequential(
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        )

    def forward(self, x):
        #print(f"[S3Dblock] Input: {x.shape}")
        x = self.conv(x)
        #print(f"[S3Dblock] Output: {x.shape}")
        return x

def _to_4d_tensor(x, depth_stride=None):
    """
    NxCxDxHxW => (N*D)xCxHxW
    Se usa para hacer 'Conv2d' en la dimensión espacial, mezclando lote y profundidad.
    """
    #print(f"[_to_4d_tensor] Input: {x.shape}")
    x = x.transpose(0, 2)  # NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]
    depth = x.size(0)
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)          # => 1x(N*D)xCxHxW
    x = x.squeeze(0)             # => (N*D)xCxHxW
    #print(f"[_to_4d_tensor] Output: {x.shape}")
    return x, depth

def _to_5d_tensor(x, depth):
    """
    Convierte de (N*D)xCxHxW => NxCxDxHxW
    """
    #print(f"[_to_5d_tensor] Input: {x.shape}")
    x = torch.split(x, depth)    # => N*[DxCxHxW]
    x = torch.stack(x, dim=0)    # => NxDxCxHxW
    x = x.transpose(1, 2)        # => NxCxDxHxW
    #print(f"[_to_5d_tensor] Output: {x.shape}")
    return x

class Block(nn.Module):
    """
    Bloque principal que mezcla 3D y 2D conv, 
    aprovechando _to_4d_tensor y _to_5d_tensor
    """
    def __init__(self, wn, n_feats, n_conv):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=False)

        # Tres secuencias 3D
        # n_feats => se asume que la 'profundidad' (en dimension C) ya es n_feats
        # Según la red original, no se define in/out canal en S3Dblock, 
        # sino que S3Dblock(wn, n_feats) => in/out = n_feats
        Block1 = []
        for _ in range(n_conv):
            Block1.append(S3Dblock(wn, n_feats))
        self.Block1 = nn.Sequential(*Block1)

        Block2 = []
        for _ in range(n_conv):
            Block2.append(S3Dblock(wn, n_feats))
        self.Block2 = nn.Sequential(*Block2)

        Block3 = []
        for _ in range(n_conv):
            Block3.append(S3Dblock(wn, n_feats))
        self.Block3 = nn.Sequential(*Block3)

        # reduceF: in_channels= n_feats*3 => out_channels= n_feats
        self.reduceF = BasicConv3d(wn, n_feats*3, n_feats, kernel_size=1, stride=1)
        self.Conv    = S3Dblock(wn, n_feats)
        self.gamma   = nn.Parameter(torch.ones(3))

        # Bloques 2D
        self.conv1 = nn.Sequential(
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)),
            self.relu,
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        )

        self.conv2 = nn.Sequential(
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)),
            self.relu,
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        )

        self.conv3 = nn.Sequential(
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)),
            self.relu,
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        )

    def forward(self, x):
        #print(f"[Block] Input: {x.shape}")
        # Bloques 3D
        x1 = self.Block1(x) + x
        x2 = self.Block2(x1) + x1
        x3 = self.Block3(x2) + x2

        # Convertir cada x en 4D => pasar Conv2d => volver 5D
        x1_4d, d1 = _to_4d_tensor(x1)
        x1_4d = self.conv1(x1_4d)
        x1_5d = _to_5d_tensor(x1_4d, d1)

        x2_4d, d2 = _to_4d_tensor(x2)
        x2_4d = self.conv2(x2_4d)
        x2_5d = _to_5d_tensor(x2_4d, d2)

        x3_4d, d3 = _to_4d_tensor(x3)
        x3_4d = self.conv3(x3_4d)
        x3_5d = _to_5d_tensor(x3_4d, d3)

        # Combinar
        x_cat = torch.cat([
            self.gamma[0] * x1_5d,
            self.gamma[1] * x2_5d,
            self.gamma[2] * x3_5d
        ], dim=1)  # Concat en canal

        #print(f"Shape after cat: {x_cat.shape}")

        x_red = self.reduceF(x_cat)
        #print(f"Shape after reduceF: {x_red.shape}")

        x_out = self.Conv(x_red)
        #print(f"Shape after last Conv3d: {x_out.shape}")

        return x_out


class MCNet(nn.Module):
    def __init__(self, args):
        super(MCNet, self).__init__()


        scale      = args.upscale_factor  # factor de escala
        n_colors   = args.n_colors        # 3 para RGB
        n_feats    = args.n_feats         # 32
        n_conv     = args.n_conv          # 1
        kernel_size= 3

        # band_mean si deseas, ajustado a 3 bandas:
        # band_mean = (0.485, 0.456, 0.406) # p.e. en RGB normal
        # O conserva tu original si no te importa
        band_mean = (0.0939, 0.0950, 0.0869)  # Ejemplo ficticio con 3 bandas
        wn = lambda x: torch.nn.utils.weight_norm(x)

        # Ajustar band_mean a 3 bandas
        self.band_mean = torch.autograd.Variable(
            torch.FloatTensor(band_mean)
        ).view([1, len(band_mean), 1, 1])

        if args.cuda:
            self.band_mean = self.band_mean.cuda()

        # 'head' conv3d: in_channel=1, out_channel= n_feats
        self.head = wn(nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size//2))

        # SSRM Bloques
        self.SSRM1 = Block(wn, n_feats, n_conv)
        self.SSRM2 = Block(wn, n_feats, n_conv)
        self.SSRM3 = Block(wn, n_feats, n_conv)
        self.SSRM4 = Block(wn, n_feats, n_conv)

        # Tail: transposed conv para upsample => conv3d => 1
        tail = [
            wn(nn.ConvTranspose3d(n_feats, n_feats,
                                  kernel_size=(3, 2+scale, 2+scale),
                                  stride=(1, scale, scale),
                                  padding=(1, 1, 1))),

            wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2))
        ]
        self.tail = nn.Sequential(*tail)

    def forward(self, x, *args):

        """
        x: [N, 3, H, W] (RGB)
        1) Quitamos band_mean
        2) unsqueeze(1) => [N, 1, 3, H, W]
        3) head => [N, n_feats, 3, H, W]
        ...
        4) tail => [N, 1, 3, H_out, W_out]
        5) squeeze => [N, 3, H_out, W_out]
        6) + band_mean
        """
        #print(f"[MCNet] Input shape: {x.shape}")
        # Ajustar la mean
        self.band_mean = self.band_mean.to(x.device)
        x = x - self.band_mean  # quitar mean canal por canal

        x = x.unsqueeze(1)     # => [N, 1, 3, H, W]
        #print(f"[MCNet] After unsqueeze(1): {x.shape}")

        T = self.head(x)       # => [N, n_feats, 3, H, W]
        #print(f"[MCNet] After head: {T.shape}")

        x = self.SSRM1(T)
        x = x + T              # => residual
        x = self.SSRM2(x)
        x = x + T
        x = self.SSRM3(x)
        x = x + T
        x = self.SSRM4(x)
        x = x + T

        #print(f"[MCNet] Before tail: {x.shape}")
        x = self.tail(x)
        #print(f"[MCNet] After tail: {x.shape}")  # => [N, 1, 3, H_out, W_out]

        x = x.squeeze(1)      # => [N, 3, H_out, W_out]
        #print(f"[MCNet] After squeeze(1): {x.shape}")

        x = x + self.band_mean.to(x.device)
        #print(f"[MCNet] Final Output: {x.shape}")

        return x

############################################
###############PROPUESTO####################
############################################
#Visual Attention Network with Large Kernel Attention (Based in Code GitHub)
# Bloque de Atención: Large Kernel Attention (LKA)
class LargeKernelAttention(nn.Module):
    def __init__(self, n_feats, kernel_size, reduction):
        super(LargeKernelAttention, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(n_feats // reduction, n_feats // reduction, kernel_size=kernel_size, padding=kernel_size // 2, groups=n_feats // reduction)
        self.conv3 = nn.Conv2d(n_feats // reduction, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return x * self.sigmoid(y)


# Bloque de Atención Cruzada entre canales RGB
class CrossChannelAttention(nn.Module):
    def __init__(self, n_feats, reduction):
        super(CrossChannelAttention, self).__init__()
        self.query = nn.Conv2d(n_feats, n_feats // reduction, 1)
        self.key = nn.Conv2d(n_feats, n_feats // reduction, 1)
        self.value = nn.Conv2d(n_feats, n_feats, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, r, g, b):
        B, C, H, W = r.size()
        rgb = torch.cat([r, g, b], dim=1)
        query = self.query(rgb).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(rgb).view(B, -1, H * W)
        value = self.value(rgb).view(B, -1, H * W)
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        r_out, g_out, b_out = torch.chunk(out, chunks=3, dim=1)
        return r + r_out, g + g_out, b + b_out

#DCANet: Dual Convolutional Neural Network with Attention for Image Blind Denoising
# Bloque de Denoising basado en DCANet
class DenoisingBlock(nn.Module):
    def __init__(self, n_feats):
        super(DenoisingBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x + residual


# Modelo Propuesto con Módulos de Atención y Denoising
class Propuesto(nn.Module):
    def __init__(self, args):
        super(Propuesto, self).__init__()
        
        upscale_factor = int(args.upscale_factor)
        n_feats = args.n_feats
        reduction = args.cross_attention.get("query_key_reduction", 8)
        
        wn = lambda x: nn.utils.weight_norm(x)

        # Cabezas para procesar canales RGB
        self.R_head = wn(nn.Conv2d(1, n_feats, kernel_size=3, stride=1, padding=1))
        self.G_head = wn(nn.Conv2d(1, n_feats, kernel_size=3, stride=1, padding=1))
        self.B_head = wn(nn.Conv2d(1, n_feats, kernel_size=3, stride=1, padding=1))

        # Bloques de Denoising
        self.R_denoising = DenoisingBlock(n_feats)
        self.G_denoising = DenoisingBlock(n_feats)
        self.B_denoising = DenoisingBlock(n_feats)

        # Bloques de Atención
        self.R_attention = LargeKernelAttention(n_feats)
        self.G_attention = LargeKernelAttention(n_feats)
        self.B_attention = LargeKernelAttention(n_feats)

        # Atención Cruzada
        self.cross_attention = CrossChannelAttention(n_feats, reduction)

        # Reconstrucción
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # Separar canales RGB
        r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]

        # Procesar cada canal por separado
        r_feat = self.R_head(r)
        g_feat = self.G_head(g)
        b_feat = self.B_head(b)

        r_feat = self.R_denoising(r_feat)
        g_feat = self.G_denoising(g_feat)
        b_feat = self.B_denoising(b_feat)

        r_feat = self.R_attention(r_feat)
        g_feat = self.G_attention(g_feat)
        b_feat = self.B_attention(b_feat)

        # Aplicar Atención Cruzada
        r_feat, g_feat, b_feat = self.cross_attention(r_feat, g_feat, b_feat)

        # Combinar características y reconstruir
        combined_feat = r_feat + g_feat + b_feat
        out = self.tail(combined_feat)

        return out

############################################
###############PROPUESTO 2##################
############################################
#MEZCLA DE ATTENTION Y 3D CONV EN MÓDULOS

