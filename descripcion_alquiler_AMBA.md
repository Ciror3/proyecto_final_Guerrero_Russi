# Descripción del Dataset: alquiler_AMBA_dev.csv

El dataset `alquiler_AMBA_dev.csv` contiene los precios de alquiler de departamentos en el Área Metropolitana de Buenos Aires (AMBA), publicados en Mercado Libre Argentina durante 2021 y 2022. Los precios de los alquileres fueron ajustados por inflación, lo que significa que se presentan en valores constantes a lo largo del tiempo. 

Para construir la base de datos, se generaron polígonos de 200m x 200m, y cada publicación fue vinculada a uno de estos polígonos. Así, en lugar de utilizar coordenadas exactas, las publicaciones fueron asociadas al centroide de su polígono correspondiente para garantizar la anonimización de los datos. La Figura 1 muestra la distribución espacial de los polígonos creados. Para una visualización más detallada, consultar el archivo `mapa_amba.html`.

## Características del Dataset

- **id_grid**: ID del polígono.
- **MesListing**: Mes en que se listó la propiedad.
- **TIPOPROPIEDAD**: Tipo de propiedad (ej. departamento, casa, etc.).
- **STotalM2**: Superficie total de la propiedad en metros cuadrados.
- **SConstrM2**: Superficie construida de la propiedad en metros cuadrados.
- **Dormitorios**: Cantidad de dormitorios en la propiedad.
- **Banos**: Cantidad de baños en la propiedad.
- **Ambientes**: Cantidad de ambientes en la propiedad.
- **SitioOrigen**: Origen del anuncio (ej. sitio web, aplicación).
- **Amoblado**: Indica si la propiedad está amoblada.
- **Antiguedad**: Antigüedad de la propiedad en años.
- **Cisterna**: Indica si la propiedad tiene cisterna.
- **AccesoInternet**: Indica si hay acceso a Internet en la propiedad.
- **BusinessCenter**: Indica si hay un centro de negocios en la propiedad.
- **Gimnasio**: Indica si hay gimnasio en la propiedad.
- **Laundry**: Indica si hay lavandería en la propiedad.
- **Calefaccion**: Indica si hay calefacción en la propiedad.
- **SalonDeUsosMul**: Indica si hay salón de usos múltiples.
- **AireAC**: Indica si hay aire acondicionado.
- **Recepcion**: Indica si hay recepción en la propiedad.
- **Estacionamiento**: Indica si hay estacionamiento.
- **Jacuzzi**: Indica si hay jacuzzi.
- **AreaJuegosInfantiles**: Indica si hay área de juegos infantiles.
- **Chimenea**: Indica si hay chimenea.
- **Ascensor**: Indica si hay ascensor en la propiedad.
- **SalonFiestas**: Indica si hay salón de fiestas.
- **Seguridad**: Indica si hay seguridad en la propiedad.
- **Pileta**: Indica si hay pileta.
- **Cocheras**: Indica si hay cocheras.
- **PistaJogging**: Indica si hay pista de jogging.
- **EstacionamientoVisitas**: Indica si hay estacionamiento para visitas.
- **Lobby**: Indica si hay lobby en la propiedad.
- **LocalesComerciales**: Indica si hay locales comerciales en la propiedad.
- **SistContraIncendios**: Indica si hay sistema contra incendios.
- **AreaParrillas**: Indica si hay área de parrillas.
- **CanchaTennis**: Indica si hay cancha de tenis.
- **AreaCine**: Indica si hay área de cine.
- **ITE_ADD_CITY_NAME**: Nombre de la ciudad donde se encuentra la propiedad.
- **ITE_ADD_STATE_NAME**: Nombre de la provincia donde se encuentra la propiedad.
- **ITE_ADD_NEIGHBORHOOD_NAME**: Nombre del barrio donde se encuentra la propiedad.
- **ITE_TIPO_PROD**: Estado de la propiedad; los valores posibles son 'U' (usado), 'N' (nuevo), y 'S' (sin clasificación).
- **LONGITUDE**: Longitud del centroide del polígono asignado a la propiedad.
- **LATITUDE**: Latitud del centroide del polígono asignado a la propiedad.
- **year**: Año en que se listó la propiedad.
- **precio_pesos_constantes**: Precio de la propiedad ajustado a pesos constantes. Target.