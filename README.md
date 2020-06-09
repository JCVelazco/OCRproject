# OCRproject
OCR project.

El proyecto cuenta hasta el momento con tres secciones diferentes

La primera es la binarización de la imagen, para lo que se usa niblack (que a su vez usa otras funciones).

La segunda sección es el búsqueda de los objetos y la medición de sus características (como los puntos que delimitan el rectángulo mas pequeño en que cabe el caracter, o el area que ocupa dentro de la imagen). Esto se hace usando objSrch2 que encuentra las regiones (o letras) usando recursión. Luego se crean las cajas que delimitan a las letras (con el uso de boxing) y se limpia la lista de letras encontradas para quitar malas mediciones que pueden ocurrir (usando boxCleaning). Finalmente, se pasa a la creación de una lista de objetos llamada "cajas", donde se pondrán las características que requiramos para estos objetos.

Finalmente, en el apartado de "efectos visuales", se añaden cosas como el dibujo de las cajas o el coloreado de las letras, a fin de poder visualizar si los algoritmos anteriores hacen lo que deben de hacer.


To run the project:

- Clone it
- cd Ocrproject
- python pruebasFinales.py

Enjoy :D
