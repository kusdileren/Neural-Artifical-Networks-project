Bu kısımdaki kodlar 3 tane ana iş yüklerini icra etmektedir. 
1)  Tek katmanlı Neuron ağ modeli ağırlıkları rastgele atanarak kurulur, pictureBox tıklanarak 2B veri seti toplanır. 
2)  Diskte kaytılı veri kümesi okunarak pictureBox üzerine yerleştirilir: 
     Veri kümesindeki dosyada format :  
               İlk satır          : Dimension, +/-Weight, +/-Height, sınıf sayısı, 
               Sonraki satırlarda :  Data + label   : verileri sıralı tutulmaktadır. 
     Network un ağırlık değerleri dosyasındaki format : 
               İlk satırda      : Layer sayısı, İnput Dimension (Veri boyutu), Sınıf Sayısı, 
               Sonraki satırda  : İlk katmandan son aktmana kadarki her katmandaki her bir nöronun input verisi için ağırlık değerleri + 
        sonrasında her katmandaki her bir nöron için bias ağırlık değerleri) diske kayıt edilmektedir,
3)  Diske veri kümesi kayıt edilmesi : Ara yüz aracılığı ile girilen samples ve network ağırlık verileri diske kayıt edilir.
      Formatı okumada açıklanan format türündedir.                     
19.10.2023

ogrenci erişimi içn şifresi :  ysa_2023