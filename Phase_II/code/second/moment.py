import numpy as np

class mom():
      """
      This class objects define moments for different order.
      """

      def mom_def(self, img):
          """
          calculate the moment of an image from 0 to 3rd order
          
          Input :
          -------
          img : the file with shape (x, y)
          
          Output :
          --------
          tuple : Monopole, centroid, 2nd and 3rd order moment
          """
    
          height, width = img.shape
          x = np.linspace(-64, 64, width)
          y = np.linspace(-64, 64, height)
          x, y = np.meshgrid(x, y)

    
          # monopole
          M00 = np.sum(img)

          # The centroid
          mx, my = np.sum(x*img)/M00, np.sum(y*img)/M00       

          # The second-order central moments
          mu11 = np.sum((x-mx) * (y-my) * img)/M00
          mu20 = np.sum((x-mx)**2 * img)/M00
          mu02 = np.sum((y-my)**2 * img)/M00

          # The third order moment
          mu30 = np.sum((x-mx)**3 * img)/M00
          mu03 = np.sum((y-my)**3 * img)/M00
          mu21 = np.sum((x-mx)**2 * (y-my) * img)/M00
          mu12 = np.sum((x-mx) * (y-my)**2 * img)/M00

          return M00, mx, my, mu11, mu20, mu02, mu30, mu03, mu21, mu12
          
      def mom_cal(self, imgG, imgP):
    
          n = imgG.shape[0]
    
          mxG, myG = [], []
          mxP, myP = [], []
    
          mu11G, mu20G, mu02G, mu30G, mu03G, mu21G, mu12G = [], [], [], [], [], [], []
          mu11P, mu20P, mu02P, mu30P, mu03P, mu21P, mu12P = [], [], [], [], [], [], []

          for i in range(n):
              M00g, mxg, myg, mu11g, mu20g, mu02g, mu30g, mu03g, mu21g, mu12g = self.mom_def(imgG[i])
              M00p, mxp, myp, mu11p, mu20p, mu02p, mu30p, mu03p, mu21p, mu12p = self.mom_def(imgP[i])

              mxG.append(mxg); myG.append(myg)
              mxP.append(mxp); myP.append(myp)

              mu11G.append(mu11g); mu20G.append(mu20g); mu02G.append(mu02g)
              mu30G.append(mu30g); mu03G.append(mu03g); mu21G.append(mu21g); mu12G.append(mu12g)

              mu11P.append(mu11p); mu20P.append(mu20p); mu02P.append(mu02p)
              mu30P.append(mu30p); mu03P.append(mu03p); mu21P.append(mu21p); mu12P.append(mu12p)
              
          mxG, myG = np.array(mxG), np.array(myG)
          mxP, myP = np.array(mxP), np.array(myP)

          mu11G, mu20G, mu02G = np.array(mu11G), np.array(mu20G), np.array(mu02G)
          mu11P, mu20P, mu02P = np.array(mu11P), np.array(mu20P), np.array(mu02P)

          mu30G, mu03G = np.array(mu30G), np.array(mu03G)
          mu30P, mu03P = np.array(mu30P), np.array(mu03P)

          mu21G, mu12G = np.array(mu21G), np.array(mu12G)
          mu21P, mu12P = np.array(mu21P), np.array(mu12P)
          
          # calculate the scatter in centroid
          Sc = n**(-1) * np.sqrt(np.sum((mxG - mxP)**2 + (myG - myP)**2))
          
          # calculate the scatter in central moment
          S11 = n**(-1) * np.sqrt(np.sum((mu11G - mu11P)**2))
          S20 = n**(-1) * np.sqrt(np.sum((mu20G - mu20P)**2))
          S02 = n**(-1) * np.sqrt(np.sum((mu02G - mu02P)**2))
          S12 = n**(-1) * np.sqrt(np.sum((mu12G - mu12P)**2))
          S21 = n**(-1) * np.sqrt(np.sum((mu21G - mu21P)**2))
          S30 = n**(-1) * np.sqrt(np.sum((mu30G - mu30P)**2))
          S03 = n**(-1) * np.sqrt(np.sum((mu03G - mu03P)**2))
          
          return Sc, S11, S20, S02, S12, S21, S30, S03
          
      def asym(self, img):

          # Rotate by 180 degrees
          img180 = img[::-1, ::-1]

          # Residual
          resi = img - img180

          # Monopole (total flux)
          M00 = np.sum(img)

          # Asymmetry statistic
          A = np.sum(np.abs(resi)) / M00

          return A
          
      def akar(self, mu11, mu20, mu02):
          """
          Calculate the shape of object in given image
          
          Input :
          -------
          mu11 : The 2nd order moment along x and y axis
          mu20 : The 2nd order moment along x axis
          mu02 : The 2nd order moment along y axis
          
          Output :
          --------
          tuple : Orientation, semi-major, semi-minor, eccentricity and area of object.
          """
    
          muplus = (mu20 + mu02)/2
          muminus = (mu20 - mu02)/2
    
          # the orientation
          alpha = np.arctan2(mu11, muminus)/2

          # the eigen-vectors
          delta = (mu11**2 + muminus**2)**(1/2)
          lam1, lam2 = muplus + delta, muplus - delta

          # the semi-major and semi-minor axis
          a, b = 2*lam1**(1/2), 2*lam2**(1/2)

          # the eccentricity
          e = (1-lam2/lam1)**(1/2)

          # the area of ellipse
          A = 4*np.pi*(mu20*mu02 - mu11**2)**(1/2)
          print('reconstructed parameter : alpha=%f, a=%f b=%f, e=%f, Area=%f' % (alpha, a, b, e, A))
    
          return alpha, a, b, e, A
          
      def ellip(self, img, mx, my, alpha, a, e):
          """
          The structure of an ellipse for given Orientation, semi-major, eccentricity.
          
          Input :
          -----
          img : It will define the size of output image (x, y)
          
          mx : The centroid of object along x 
          
          my : The centroid of object along y
          
          alpha : The orientation of object
          
          a : The semi-major axis
           
          e : The eccentricity
          
          Output :
          -------
          metric : 2-d metric of structure of object
          """
      
          height, width = img.shape
          x = np.arange(0, width)
          y = np.arange(0, height)
          x, y = np.meshgrid(x, y)
    
          xp,yp = x-mx,y-my
          cs,sn = np.cos(alpha),np.sin(alpha)
          xp,yp = cs*xp + sn*yp, -sn*xp + cs*yp
          w = 0*x
          w[xp**2 + yp**2/(1-e*e) < a*a] = 1

          return w
          
      def prakar(self, mx, my, alpha, a, b):
          """
          The shape and size of object
          
          Input :
          -------
          mx : The centroid of object along x 
          
          my : The centroid of object along y
          
          alpha : The orientation of object
          
          a : The semi-major axis
           
          b : The semi-minor axis
          
          Output :
          -------
          tuple : array of x and y
          """
          t = np.linspace(0, 2*np.pi, 501)
    
          X = a * np.cos(t)
          Y = b * np.sin(t)
    
          x = X * np.cos(alpha) - Y * np.sin(alpha) + mx
          y = X * np.sin(alpha) + Y * np.cos(alpha) + my
    
          return x, y
          
          
