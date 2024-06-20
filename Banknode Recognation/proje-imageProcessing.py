# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np

resim_dizini = 'C:/Users/muham/.spyder-py3/Orijinal'
orijinal_resimler = [dosya for dosya in os.listdir(resim_dizini) if dosya.endswith(('.png', '.jpg', '.jpeg'))]

def islenmis_resmi_kaydet(islenmis_resim, orijinal_resim_ad, islem_turu):
    islenmis_resim_dizini = 'C:/Users/muham/.spyder-py3/İslenmis'
    islenmis_resim_ad = f'{orijinal_resim_ad}_{islem_turu}.jpg'
    kayit_yolu = os.path.join(islenmis_resim_dizini,'/', islenmis_resim_ad)
    cv2.imwrite(kayit_yolu, islenmis_resim)

def yirtik_banknot_efekti(resim):
    height, width, _ = resim.shape
    yirtik_resim = resim.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for _ in range(yirtiklik_derecesi):
        x = random.randint(0, width)
        y = random.randint(0, height)
        cv2.circle(mask, (x, y), random.randint(2, 10), (255), -1)

    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    yirtik_resim = cv2.bitwise_and(yirtik_resim, yirtik_resim, mask=mask)  
    return yirtik_resim

def rastgele_bant_ekle(resim):
    yukseklik, genislik, _ = resim.shape

    bant_genislik = np.random.randint(50, 100)
    bant_yukseklik = np.random.randint(50, 100)

    baslangic_x = np.random.randint(0, genislik - bant_genislik)
    baslangic_y = np.random.randint(0, yukseklik - bant_yukseklik)
    sarimtirak_renk = (0, 225, 255)

    bant = np.full((bant_yukseklik, bant_genislik, 3), sarimtirak_renk, dtype=np.uint8)
    bant = cv2.addWeighted(resim[baslangic_y:baslangic_y+bant_yukseklik, baslangic_x:baslangic_x+bant_genislik], 1, bant, 1, 0)

    bantli_resim[baslangic_y:baslangic_y+bant_yukseklik, baslangic_x:baslangic_x+bant_genislik] = bant
    return bantli_resim

def kursun_veya_tukenmezle_yazilmis_efekti(resim):
    height, width, _ = resim.shape
    drawn_resim = resim.copy()

    for _ in range(line_count): 
        thickness = random.randint(1, 2)
        
        p1 = (random.randint(0, width), random.randint(0, height))
        p2 = (random.randint(0, width), random.randint(0, height))
        p3 = (random.randint(0, width), random.randint(0, height))
        p4 = (random.randint(0, width), random.randint(0, height))
        
        t = np.linspace(0, 1, 100)
        x = (1-t)**3 * p1[0] + 3*(1-t)**2 * t * p2[0] + 3*(1-t) * t**2 * p3[0] + t**3 * p4[0]
        y = (1-t)**3 * p1[1] + 3*(1-t)**2 * t * p2[1] + 3*(1-t) * t**2 * p3[1] + t**3 * p4[1]

        curve_points = np.column_stack((x, y)).astype(int)
        for i in range(len(curve_points) - 1):
            cv2.line(drawn_resim, tuple(curve_points[i]), tuple(curve_points[i+1]), color, thickness)
            
    return drawn_resim

for dosya_ad in orijinal_resimler:

    resim_yolu = os.path.join(resim_dizini, dosya_ad)
    dosya, uzantı = dosya_ad.rsplit('.', 1)
    resim = cv2.imread(resim_yolu)
    islenmis_resmi_kaydet(resim, dosya, 'Original')
    
    donmus_resim = cv2.rotate(resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_resim, dosya, 'Rotated Original')
    
    ###########################################################################################################
    
    yirtiklik_derecesi = 700
    az_yirtik_resim = resim.copy()
    az_yirtik_resim = yirtik_banknot_efekti(az_yirtik_resim)
    islenmis_resmi_kaydet(az_yirtik_resim, dosya, 'Slightly Ripped 0')
    
    donmus_az_yirtik_resim = cv2.rotate(az_yirtik_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_az_yirtik_resim, dosya, 'Rotated Slightly Ripped 0')
    
    ###########################################################################################################
    
    yirtiklik_derecesi = 500
    cok_yirtik_resim = resim.copy()
    cok_yirtik_resim = yirtik_banknot_efekti(cok_yirtik_resim)
    islenmis_resmi_kaydet(cok_yirtik_resim, dosya, 'Very Ripped 0')
    
    donmus_cok_yirtik_resim = cv2.rotate(cok_yirtik_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_cok_yirtik_resim, dosya, 'Rotated Very Ripped 0')
    
    ###########################################################################################################
    
    bantli_resim = resim.copy()
    bant_sayisi = random.randint(1,2)
    while (bant_sayisi>0):
        bantli_resim = rastgele_bant_ekle(bantli_resim)
        bant_sayisi = bant_sayisi-1
        
    islenmis_resmi_kaydet(bantli_resim, dosya, 'Slightly Banded 0')
    donmus_bantli_resim = cv2.rotate(bantli_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_bantli_resim, dosya, 'Rotated Slightly Banded 0')
    
    ###########################################################################################################
    
    bantli_resim = resim.copy()
    bant_sayisi = random.randint(2,4)
    while (bant_sayisi>0):
        bantli_resim = rastgele_bant_ekle(bantli_resim)
        bant_sayisi = bant_sayisi-1
        
    islenmis_resmi_kaydet(bantli_resim, dosya, 'Very Banded 0')
    donmus_bantli_resim = cv2.rotate(bantli_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_bantli_resim, dosya, 'Rotated Very Banded 0')
    
    ###########################################################################################################
    
    line_count, color = 5, (0, 0, 0)
    az_kursun_cizilmis_resim = resim.copy()
    az_kursun_cizilmis_resim = kursun_veya_tukenmezle_yazilmis_efekti(az_kursun_cizilmis_resim)
    islenmis_resmi_kaydet(az_kursun_cizilmis_resim, dosya, 'Slightly Drawn In Pencil 0')
    
    donmus_az_kursun_cizilmis_resim = cv2.rotate(az_kursun_cizilmis_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_az_kursun_cizilmis_resim, dosya, 'Rotated Slightly Drawn In Pencil 0')
    
    ###########################################################################################################
    
    line_count, color = 15, (0, 0, 0)
    cok_kursun_cizilmis_resim = resim.copy()
    cok_kursun_cizilmis_resim = kursun_veya_tukenmezle_yazilmis_efekti(cok_kursun_cizilmis_resim)
    islenmis_resmi_kaydet(cok_kursun_cizilmis_resim, dosya, 'Very Drawn In Pencil 0')
    
    donmus_cok_kursun_cizilmis_resim = cv2.rotate(cok_kursun_cizilmis_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_cok_kursun_cizilmis_resim, dosya, 'Rotated Very Drawn In Pencil 0')
    
    ###########################################################################################################
    
    line_count, color = 5, (128, 0, 0)
    az_tukenmez_cizilmis_resim = resim.copy()
    az_tukenmez_cizilmis_resim = kursun_veya_tukenmezle_yazilmis_efekti(az_tukenmez_cizilmis_resim)
    islenmis_resmi_kaydet(az_tukenmez_cizilmis_resim, dosya, 'Slightly Drawn In Pen 0')
    
    donmus_az_tukenmez_cizilmis_resim = cv2.rotate(az_tukenmez_cizilmis_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_az_tukenmez_cizilmis_resim, dosya, 'Rotated Slightly Drawn In Pen 0')
    
    ###########################################################################################################
    
    line_count, color = 15, (128, 0, 0)
    cok_tukenmez_cizilmis_resim = resim.copy()
    cok_tukenmez_cizilmis_resim = kursun_veya_tukenmezle_yazilmis_efekti(cok_tukenmez_cizilmis_resim)
    islenmis_resmi_kaydet(cok_tukenmez_cizilmis_resim, dosya, 'Very Drawn In Pen 0')
    
    donmus_cok_tukenmez_cizilmis_resim = cv2.rotate(cok_tukenmez_cizilmis_resim, cv2.ROTATE_90_CLOCKWISE)
    islenmis_resmi_kaydet(donmus_cok_tukenmez_cizilmis_resim, dosya, 'Rotated Very Drawn In Pen 0')
    