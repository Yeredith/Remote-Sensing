<<<<<<< HEAD
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
=======

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TwoCNN(nn.Module):
    def __init__(self, wn, n_feats=64): 
        super(TwoCNN, self).__init__()

        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3), stride=1, padding=(1,1)))
               
    def forward(self, x):
    
        out = self.body(x)
        out = torch.add(out, x)
        
        return out             
>>>>>>> ef3cf9e75d1c1da1797a6d08e5d3a5d4c332c368

class ThreeCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(ThreeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)
<<<<<<< HEAD
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
=======

        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))

        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))            

        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)

    def forward(self, x): 
        out = x
        for i in range(2):
              
            out = torch.add(self.body_spatial[i](out), self.body_spectral[i](out))
            if i == 0:
                out = self.act(out)
    
        out = torch.add(out, x)        
        return out
                                                                                                                                                                                                            
class SFCSR(nn.Module):
    def __init__(self, args):
        super(SFCSR, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
 
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
                                                        
        ThreeHead = []
        ThreeHead.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        ThreeHead.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))  
        self.ThreeHead = nn.Sequential(*ThreeHead)
        

        TwoHead = []
        TwoHead.append(wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        self.TwoHead = nn.Sequential(*TwoHead)

        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1))))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1))))
            TwoTail.append(nn.PixelShuffle(3))  

        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1))))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)
                        	 
        twoCNN = []
        for _ in range(self.n_module):
            twoCNN.append(TwoCNN(wn, n_feats))
        self.twoCNN = nn.Sequential(*twoCNN)
        
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))
        
        threeCNN = []
        for _ in range(self.n_module):
            threeCNN.append(ThreeCNN(wn, n_feats))
        self.threeCNN = nn.Sequential(*threeCNN)
      
        reduceD = []
        for _ in range(self.n_module):
            reduceD.append(wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1)) )
        self.reduceD = nn.Sequential(*reduceD)
                                  
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1))   
        
        threefusion = []               
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))          
        self.threefusion = nn.Sequential(*threefusion)
        

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1)) 
        
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1))    
        
    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         
        
        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y
        
        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats  
>>>>>>> ef3cf9e75d1c1da1797a6d08e5d3a5d4c332c368
    
###########################
           #MCNET
###########################

class BasicConv3d(nn.Module):
    def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0,0,0)):
        super(BasicConv3d, self).__init__()
<<<<<<< HEAD
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

=======
        self.conv = wn(nn.Conv3d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding))
        # Cambiado inplace=True a inplace=False
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class S3Dblock(nn.Module):
    def __init__(self, wn, n_feats):
        super(S3Dblock, self).__init__()

        self.conv = nn.Sequential(
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        )            
       
    def forward(self, x): 
    	   	
        return self.conv(x)

def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x
    
    
class Block(nn.Module):
    def __init__(self, wn, n_feats, n_conv):
        super(Block, self).__init__()

        # Cambiado inplace=True a inplace=False
        self.relu = nn.ReLU(inplace=False)
        
        Block1 = []  
        for i in range(n_conv):
            Block1.append(S3Dblock(wn, n_feats)) 
        self.Block1 = nn.Sequential(*Block1)         

        Block2 = []  
        for i in range(n_conv):
            Block2.append(S3Dblock(wn, n_feats)) 
        self.Block2 = nn.Sequential(*Block2) 
        
        Block3 = []  
        for i in range(n_conv):
            Block3.append(S3Dblock(wn, n_feats)) 
        self.Block3 = nn.Sequential(*Block3) 
        
        self.reduceF = BasicConv3d(wn, n_feats*3, n_feats, kernel_size=1, stride=1)                                                            
        self.Conv = S3Dblock(wn, n_feats)
        self.gamma = nn.Parameter(torch.ones(3))   
         
        conv1 = []   
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv1.append(self.relu)
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv1 = nn.Sequential(*conv1)           

        conv2 = []   
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv2.append(self.relu)
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv2 = nn.Sequential(*conv2)  
        
        conv3 = []   
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv3.append(self.relu)
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv3 = nn.Sequential(*conv3)          
                                                          
    def forward(self, x): 
        res = x
        x1 = self.Block1(x) + x 
        x2 = self.Block2(x1) + x1         
        x3 = self.Block3(x2) + x2     

        x1, depth = _to_4d_tensor(x1, depth_stride=1)  
        x1 = self.conv1(x1)       
        x1 = _to_5d_tensor(x1, depth)  
                             
        x2, depth = _to_4d_tensor(x2, depth_stride=1)  
        x2 = self.conv2(x2)       
        x2 = _to_5d_tensor(x2, depth)         
   
        x3, depth = _to_4d_tensor(x3, depth_stride=1)  
        x3 = self.conv3(x3)       
        x3 = _to_5d_tensor(x3, depth)  
                
        x = torch.cat([self.gamma[0]*x1, self.gamma[1]*x2, self.gamma[2]*x3], 1)                 
        x = self.reduceF(x) 
        x = self.relu(x)
        x = x + res        
        
        x = self.Conv(x)                                                                                                               
        return x  
>>>>>>> ef3cf9e75d1c1da1797a6d08e5d3a5d4c332c368

class MCNet(nn.Module):
    def __init__(self, args):
        super(MCNet, self).__init__()

<<<<<<< HEAD
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
=======
        scale = args.upscale_factor
        n_colors = args.n_colors
        n_feats = args.n_feats          
        n_conv = args.n_conv
        kernel_size = 3

        band_mean = (0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834, 
                     0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194, 
                     0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541)  # CAVE

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, 31, 1, 1])

        if args.cuda:  
            self.band_mean = self.band_mean.cuda()                    

        self.head = wn(nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size//2))        
        self.SSRM1 = Block(wn, n_feats, n_conv)              
        self.SSRM2 = Block(wn, n_feats, n_conv) 
        self.SSRM3 = Block(wn, n_feats, n_conv)           
        self.SSRM4 = Block(wn, n_feats, n_conv)  
        tail = [
            wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale,2+scale), stride=(1,scale,scale), padding=(1,1,1))),
>>>>>>> ef3cf9e75d1c1da1797a6d08e5d3a5d4c332c368
            wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2))
        ]
        self.tail = nn.Sequential(*tail)

    def forward(self, x, *args):
<<<<<<< HEAD
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
    def __init__(self, n_feats, kernel_size=15, reduction=8):
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
    def __init__(self, n_feats, reduction=8):
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
=======
        """Ajustado para ignorar argumentos adicionales."""
        self.band_mean = self.band_mean.to(x.device)
        x = x - self.band_mean  
        x = x.unsqueeze(1)
        T = self.head(x)
        
        x = self.SSRM1(T)
        x = torch.add(x, T)
        
        x = self.SSRM2(x)
        x = torch.add(x, T)
        
        x = self.SSRM3(x)
        x = torch.add(x, T)
        
        x = self.SSRM4(x)
        x = torch.add(x, T)
        
        x = self.tail(x)
        x = x.squeeze(1)
        
        x = x + self.band_mean.to(x.device)
        
        return x

#################################
           #SFCCBAM
#################################
# Bloque de Atención de Canal con regularización L2
class ChannelAttention(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Regularización L2 aplicada en cada convolución del bloque de atención
        self.fc1 = nn.Conv3d(n_feats, n_feats // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(n_feats // ratio, n_feats, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x

# Bloque CBAM simplificado (solo canal)
class CBAM(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(n_feats, ratio)

    def forward(self, x):
        x = self.channel_attention(x)
        return x

# Clase TwoCNN con regularización y CBAM simplificado
class SpatialCNN(nn.Module):
    def __init__(self, wn, n_feats=64): 
        super(SpatialCNN, self).__init__()
        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3), stride=1, padding=(1,1), bias=False))
        
        # Bloque CBAM simplificado
        self.cbam = CBAM(n_feats)
        
        # Capa de ajuste de canales, se configura dinámicamente en forward
        self.adjust_channels = None

    def forward(self, x):
        out = self.body(x)
        
        # Pasamos `out` a través del CBAM y ajustamos los canales si es necesario
        out = self.cbam(out.unsqueeze(2)).squeeze(2)  # Adaptamos CBAM a 2D convolución

        # Configuración dinámica de `adjust_channels` para igualar canales de `x`
        if out.shape[1] != x.shape[1]:
            self.adjust_channels = nn.Conv2d(out.shape[1], x.shape[1], kernel_size=1).to(out.device)
            out = self.adjust_channels(out)
        
        # Realizamos la suma
        out = torch.add(out, x)
        return out             

# Clase ThreeCNN con regularización y CBAM simplificado
class SpectralCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(SpectralCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)

        # Inicializamos las capas de convolución 3D con n_feats canales
        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)))
        
        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)))
        
        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)
        
        # Bloque CBAM simplificado
        self.cbam = CBAM(n_feats)
        
        # Ajuste de canales: se configura dinámicamente en forward() según el tamaño de entrada
        self.adjust_channels = None

    def forward(self, x): 
        out = x
        for i in range(2):  
            out_spatial = self.body_spatial[i](out)
            out_spectral = self.body_spectral[i](out)
            
            # Sumamos out_spatial y out_spectral
            out = torch.add(out_spatial, out_spectral)
            if i == 0:
                out = self.act(out)
        
        out = self.cbam(out)
        
        # Configuración dinámica de self.adjust_channels para igualar canales de `x`
        if out.shape[1] != x.shape[1]:
            self.adjust_channels = nn.Conv3d(out.shape[1], x.shape[1], kernel_size=1).to(out.device)
            out = self.adjust_channels(out)
        
        # Verificamos la compatibilidad de las dimensiones antes de la suma final con x
        if out.shape == x.shape:
            out = torch.add(out, x)
        else:
            print(f"Dimension mismatch before final addition: out {out.shape}, x {x.shape}")
            return None
        
        return out

# Clase SFCCBAM
class SFCCBAM(nn.Module):
    def __init__(self, args):
        super(SFCCBAM, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
    
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
        # Head 
        ThreeHead = [wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                     wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.ThreeHead = nn.Sequential(*ThreeHead)

        TwoHead = [wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))]
        self.TwoHead = nn.Sequential(*TwoHead)

        # Tail 
        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
            TwoTail.append(nn.PixelShuffle(3))  
        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False)))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)

        # Convoluciones y atenciones
        self.twoCNN = nn.Sequential(*[SpatialCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1, bias=False))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))

        self.threeCNN = nn.Sequential(*[SpectralCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD = nn.Sequential(*[wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False)) for _ in range(self.n_module)])                              
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1, bias=False))
        
        threefusion = [wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                       wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.threefusion = nn.Sequential(*threefusion)

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False)) 
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False))    
    
    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         

        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y

        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats  



#################################
#######Hybrid-SFCSR##############
#################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(n_feats, n_feats // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(n_feats // ratio, n_feats, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return x * self.sigmoid(scale)


# Bloque de Atención de Canal con regularización L2
class ChannelAttention_F(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(ChannelAttention_F, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(n_feats, n_feats // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(n_feats // ratio, n_feats, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention_F(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_F, self).__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=(0, padding, padding), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM_F(nn.Module):
    def __init__(self, n_feats, ratio=8, kernel_size=7):
        super(CBAM_F, self).__init__()
        self.channel_attention = ChannelAttention_F(n_feats, ratio)
        self.spatial_attention = SpatialAttention_F(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
    
class HybridAttention(nn.Module):
    """Combina CBAM y SE en un único bloque de atención híbrida."""
    def __init__(self, n_feats, ratio=8, kernel_size=7):
        super(HybridAttention, self).__init__()
        self.cbam = CBAM_F(n_feats, ratio , kernel_size )
        self.se = SqueezeExcitation(n_feats, ratio)

    def forward(self, x):
        x = self.cbam(x)  # Primero aplicar CBAM
        x = self.se(x)    # Luego aplicar SE
        return x

class ResidualDenoisingBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3):
        super(ResidualDenoisingBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

class SpaCNN(nn.Module):
    def __init__(self, wn, n_feats=64): 
        super(SpaCNN, self).__init__()
        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3), stride=1, padding=(1,1)))
        #self.cbam = CBAM(n_feats)
        #self.se = SqueezeExcitation(n_feats)
        self.hybrid_attention = HybridAttention(n_feats)
        self.denoising_block = ResidualDenoisingBlock(n_feats)

    def forward(self, x):
        out = self.body(x)
        #out = self.cbam(out.unsqueeze(2)).squeeze(2)  # CBAM adaptado para 2D
        #out = self.se(out.unsqueeze(2)).squeeze(2)  # SE adaptado para 2D
        out = self.hybrid_attention(out.unsqueeze(2)).squeeze(2)
        out = self.denoising_block(out)
        out = torch.add(out, x)
        return out


class SpeCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(SpeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)

        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))

        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))            

        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)
        #self.cbam = CBAM(n_feats)
        #self.se = SqueezeExcitation(n_feats)
        self.hybrid_attention = HybridAttention(n_feats)  # Bloque híbrido

    def forward(self, x): 
        out = x
        for i in range(2):
            out = torch.add(self.body_spatial[i](out), self.body_spectral[i](out))
            if i == 0:
                out = self.act(out)
        #out = self.cbam(out)
        #out = self.se(out)
        out = self.hybrid_attention(out)
        out = torch.add(out, x)        
        return out


class HYBRID_SE_CBAM(nn.Module):
    def __init__(self, args):
        super(HYBRID_SE_CBAM, self).__init__()
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
    
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
        # Head 
        ThreeHead = [wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                     wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.ThreeHead = nn.Sequential(*ThreeHead)

        TwoHead = [wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))]
        self.TwoHead = nn.Sequential(*TwoHead)

        # Tail 
        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
            TwoTail.append(nn.PixelShuffle(3))  
        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False)))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)

        # Convoluciones y atenciones
        self.twoCNN = nn.Sequential(*[SpaCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1, bias=False))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))

        self.threeCNN = nn.Sequential(*[SpeCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD = nn.Sequential(*[wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False)) for _ in range(self.n_module)])                              
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1, bias=False))
        
        threefusion = [wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                       wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.threefusion = nn.Sequential(*threefusion)

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False)) 
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False))    
    
    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         

        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y

        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats  
>>>>>>> ef3cf9e75d1c1da1797a6d08e5d3a5d4c332c368
