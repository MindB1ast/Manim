#manim -pql Untitled-1.py TextToImageArchitecture
from manim import *
import os

import numpy

class TextToImageArchitecture(Scene):
    def construct(self):
        # Заголовок презентации
        title_1 = Text("Демонстрация архитектур", font_size=64)
        title_2 = Text("text-to-image", font_size=64)
        title_1.to_edge(UP)
        title_2.next_to(title_1, DOWN, buff=0.5)
        title_1.move_to(np.array([0, 3, 0])) # Центрирование первой строки по горизонтали
        title_2.move_to(np.array([0, 2, 0])) # Центрирование второй строки по горизонтали
        self.play(Write(title_1), Write(title_2))
        self.wait(1)

        # Описание первой архитектуры
        architecture_1 = Text("На примере Stable diffusion", font_size=48)
        architecture_1.next_to(title_2, DOWN, buff=1)
        self.play(FadeIn(architecture_1, shift=UP))
        self.wait(1)


        # Переход к следующему слайду
        self.play(FadeOut(architecture_1,title_2, title_1 ))
        self.wait(1)

        # Заготовка для текста на втором слайде
        text_intro = Text("Предназначение", font_size=48)
        text_intro.move_to([-3.5, 3.5, 0])
        self.play(FadeIn(text_intro, shift=UP))
        title_1= Text("Модель Stable diffusion позволяет генерировать", font_size=32)
        title_2=Text("изображения с помощью текствового описания", font_size=32)
        title_1.next_to(text_intro, DOWN)
        title_1.align_to(text_intro, LEFT)
        
        title_2.next_to(title_1, DOWN)
        self.add(title_1)
        self.add(title_2)
        self.play(FadeIn(title_1, title_2, shift=UP))
        self.wait(1)

        
        Promnt_box= Square()
        Promt_title=Text('Промт', font_size=32)
        Promnt_box.move_to(DOWN)
        Promnt_box.align_to(title_2, LEFT)
        Promt_title.next_to(Promnt_box, UP)
        promt = Text("Austronaut riding\nhorse on mars", font_size=16) 
        promt.next_to(Promnt_box, direction=ORIGIN)
        promt.align_to(Promnt_box, LEFT+UP)
        group = VGroup(Promnt_box, promt, Promt_title)
        arrow = Arrow().next_to(group, RIGHT, buff=1.5)
        self.play(FadeIn(group, arrow,  shift=UP))



        image_folder = "intermediate"  # Замените на путь к вашей папке с изображениями
        images = os.listdir(image_folder)
        prev_image = None
        for image_file in images:
            if prev_image is not None:
                self.remove(prev_image)
            image_path = os.path.join(image_folder, image_file)
            image = ImageMobject(image_path).next_to(Promnt_box, RIGHT, buff=4)
            self.add(image)
            self.wait(0.2)
            prev_image = image

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        

class ModelContent(Scene):
    def construct(self):
        title_1 = Text("Из чего состоит Stable diffusion", font_size=64)
        title_1.to_edge(UP)
        image = ImageMobject('spaces_-LIA3amopGH9NC6Rf0mA_uploads_git-blob-173bc4d742d9df6aecc82e380c7ab563c08c3c80_stable-diffusion-architecture.png')
        image.next_to(title_1, DOWN)  # Разместите изображение под текстом
        #self.play(FadeIn(image))
        self.add(image)

        self.play(Write(title_1))
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
        
        title_1=Text('CLIP').shift((LEFT*2+UP)*2)
        image= ImageMobject('cma-clip-architecture.png')
        image.next_to(title_1, DOWN)
        image.scale(0.6)
        title_2=Text('Unet').shift(DOWN*2)
        image2= ImageMobject('unet.png').scale(0.15)
        image2.next_to(title_2, UP, buff=2)

        title_3=Text('VAE').shift((RIGHT*2+UP)*2)
        image3= ImageMobject('vae.png').scale(0.2)
        image3.next_to(title_3, DOWN)
        title_4=Text('3 главных компонента Stable diffusion').to_edge(UP).shift(UP*0.5)
        self.play(FadeIn(title_1, title_2, title_3, title_4, image, image2, image3, shift=UP))
        self.wait(2)
        self.play(FadeOut(title_2, title_3, title_4, image2, image3, shift=DOWN))
        self.wait(1)
        self.play(title_1.animate.scale(2).to_corner(UP+LEFT), image.animate.scale(2).move_to(ORIGIN))
        #self.play()
        self.wait(1)
        self.play(FadeOut(image, shift=DOWN))
        description=Text('CLIP (Contrastive Language-Image Pre-training)\nМодель позволяет определять на сколько хорошо соответсвует\nизображение текстовому описанию', font_size=32)
        description.next_to(title_1, DOWN).align_to(title_1, LEFT)
        Promt_box= Square().next_to(description, DOWN).align_to(description, LEFT)
        text_in_box=Text('[Кот], [Собака]', font_size=18).move_to(Promt_box, ORIGIN)#.align_to(Promt_box, ORIGIN)
        Cat_Image=ImageMobject('britanskaya-korotkosherstnaya-koshka-opisanie-porody.jpg').scale(0.25).next_to(Promt_box, DOWN )
        text_arrow = Arrow().next_to(Promt_box, RIGHT, buff=1.1)
        text_encoder_box=Square().scale(1.1).next_to(text_arrow, RIGHT, buff=0.5)
        text_in_encoder_box=Text('Текстовый энкодер', font_size=17).move_to(text_encoder_box, ORIGIN).align_to(text_encoder_box, LEFT)
        text_encoder=Group(text_encoder_box, text_in_encoder_box)
        image_arrow = Arrow().next_to(Cat_Image, RIGHT, buff=1.5).align_to(text_arrow, LEFT)
        image_encoder_box=Square().scale(1.1).next_to(image_arrow, RIGHT, buff=0.5)
        text_in_image_encoder_box=Text('Энкодер изображения', font_size=17).move_to(image_encoder_box, ORIGIN).align_to(image_encoder_box, LEFT)
        Image_encoder=Group(image_encoder_box, text_in_image_encoder_box)
        model_box=Square().next_to(((Image_encoder.get_center() + text_encoder.get_center()) / 2), buff=2.25)
        model_text=Text('Слои модели', font_size=17).move_to(model_box, ORIGIN)
        model=Group(model_box, model_text)
        text_encoder_arrow=Arrow().scale(0.6).next_to(text_encoder, RIGHT, buff=0.1).rotate(-PI/6)
        image_encoder_arrow=Arrow().scale(0.6).next_to(Image_encoder, RIGHT).align_to(text_encoder_arrow, LEFT).rotate(PI/6)
        model_arrow=Arrow().scale(0.5).next_to(model, RIGHT)

        model_arrow_text=Text('Softmax', font_size=15).next_to(model_arrow, UP)


        answer_box=Square().next_to(model_arrow, RIGHT, buff=0.1)
        answer_box_text=Text('[0.95, 0.05]',font_size=17).move_to(answer_box)
        answer_label=Text('Вероятность\nпринодлежности\nк описанию', font_size=15).next_to(answer_box, UP)
        answer=Group(answer_box, answer_box_text, answer_label)

        self.play(FadeIn(description, Promt_box, text_in_box, Cat_Image, text_arrow,image_arrow, text_encoder,Image_encoder, text_encoder_arrow, image_encoder_arrow, model, model_arrow, model_arrow_text, answer), shift=UP)
        self.wait(3)
        description2=Text('CLIP в модели Stable diffusion используется в качестве текстового\nэнкодера для получения принзаков генерируемового изображения по описанию', font_size=32).move_to(description).align_to(description, LEFT+UP)
        #self.play( shift=DOWN)
        #self.play(ReplacementTransform(description, description2))
        
        TensorBox= Square().scale(1.1).move_to(image_encoder_box)
        self.add(TensorBox)
        TensorBox_text=Text('[1.2, 5.0, 0.87, 3, ...]', font_size=18).move_to(TensorBox).align_to(TensorBox)
        Tensor=Group(TensorBox_text, TensorBox)

        self.remove(image_encoder_box)
        alls= Group(Promt_box, text_in_box, text_arrow,  text_encoder, text_encoder_arrow, model, Tensor)
        
        

        self.play(description.animate.become(description2), FadeOut(answer, model_arrow_text, model_arrow, Cat_Image, image_arrow, text_in_image_encoder_box),  image_encoder_arrow.animate.shift(RIGHT).rotate(PI), 
                  alls.animate.shift(RIGHT) )
        self.wait(3)
        description2=Text('В зависимости от генеративной модели вместо модели CLIP\nмогут быть использованы другие трансформеры', font_size=32).move_to(description).align_to(description, LEFT+UP)

        self.play( description.animate.become(description2))
        self.wait(3)
        description2=Text('Таким образом данная часть модели является не только\nуправляющей частью, но и "мозгами" модели позволяя\nотделять концепты друг от друга', font_size=32).move_to(description).align_to(description, LEFT+UP)

        self.play(FadeOut(alls, image_encoder_arrow), description.animate.shift(DOWN).become(description2))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class ClipFor15(Scene):
    def construct(self):
        title_1=Text('Как работает CLIP', font_size=36).to_corner(UP+LEFT)
        self.play(FadeIn(title_1))
        self.wait(1)
        
        description=Text('Для Stable diffusion 1.5 используется трансформер clip-ViT-L-14 с фиксированым\nразмером промта 77, 2 из которых занимают стартовый и конечный токен', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        PromtBox=Square().next_to(description, DOWN).align_to(description, LEFT)
        PromtText=Text('Austronaut riding\nhorse on mars', font_size=18).move_to(PromtBox).align_to(PromtBox, LEFT+UP)
        Promt=Group(PromtBox, PromtText)
        self.play(FadeIn(description, Promt))
        self.wait(2)
        description2=Text('Токенизатор для clip-ViT-L-14 имеет размер словаря равный 49408\nтокенов', font_size=32).move_to(description).align_to(description, LEFT+UP)
        tokenizer_arrow= Arrow().next_to(Promt, RIGHT)
        Tokenized_Box=Square().next_to(tokenizer_arrow, RIGHT)
        Tokenized_text=Text('[[49406,  2518,\n521,12343,\n6765,4558,\n525,7496,\n49407]]', font_size=18).move_to(Tokenized_Box).align_to(Tokenized_Box, LEFT+UP)
        Tokenized=Group(Tokenized_Box, Tokenized_text)
        tokenizer_arrow_text=Text('Токенизатор', font_size=18).next_to(tokenizer_arrow, UP)
        self.play(FadeIn(tokenizer_arrow, Tokenized, tokenizer_arrow_text), description.animate.become(description2))
        self.wait(2)
        description2=Text('Текстовые токены передаются в слой вложений,\nкоторый преобразует каждый токен в соответствующий\nвектор вложения, а так же добавляется значение позиционных\nэмбендингов.', font_size=32).move_to(description2).align_to(description2, LEFT+UP)
        embeddings_arrow= Arrow().next_to(Tokenized, RIGHT)
        Embeddings_Box=Square().next_to(embeddings_arrow, RIGHT)
        Embeddings_text=Text('[768],\n[768],\n[768]...', font_size=18).move_to(Embeddings_Box).align_to(Embeddings_Box, LEFT+UP)
        Embeddings=Group(Embeddings_Box, Embeddings_text)
        embeddings_arrow_text=Text('Слой вложений', font_size=18).next_to(embeddings_arrow, UP)
        alls=Group(embeddings_arrow, Embeddings, embeddings_arrow_text, tokenizer_arrow, Tokenized, tokenizer_arrow_text, Promt)
        
        self.play(description.animate.become(description2), FadeIn(embeddings_arrow, Embeddings, embeddings_arrow_text), alls.animate.shift(DOWN))
        self.wait(3)
        description2=Text('После создания вложений данные передаются в текстовый\nтрансформер с механизмом внимания, который позволяет модели\n учесть контекст каждого токена.', font_size=32).move_to(description2).align_to(description2, LEFT+UP)
        self.play(description.animate.become(description2), alls.animate.shift(DOWN))
        self.wait(3)
        description2=Text('В результате, модель CLIP выдает векторы вложений, которые\nпредставляют семантическое содержание входного текста.', font_size=32).move_to(description2).align_to(description2, LEFT+UP)
        Heatmap=ImageMobject('grid_daam-0000.png').scale(0.65).shift(DOWN)
        self.play(description.animate.become(description2), FadeOut(alls), FadeIn(Heatmap))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

class VAE(Scene):
    def construct(self):
        title_1=Text('VAE', font_size=36).to_corner(UP+LEFT)
        self.play(FadeIn(title_1))
        self.wait(1)
        description=Text('Пришло время разобрать генеративную часть Stalbe diffusion\nначнем с вариационого автоэнкодера(VAE)', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        Vae_image=ImageMobject('vae.png').scale(0.5)
        self.play(FadeIn(description, Vae_image))
        self.wait(3)
        description2=Text('VAE использвует для того что бы сжимать\nи расжимать данные в скрытое пространство\nв нашем случае сжатие изображения при обучении\nи расжатии из скрытого пространства при генерации', font_size=32).move_to(description).align_to(description, LEFT+UP)
        Gen_Image=ImageMobject('0000034-1045145337-Austronaut riding horse on mars.png').next_to(description, DOWN).align_to(description, LEFT).shift(DOWN)
        Image_in_latent_arrow= Arrow().next_to(Gen_Image, RIGHT)
        Image_in_latent_Box=Square().next_to(Image_in_latent_arrow, RIGHT)
        Image_in_latent_text=Text('[[49406,  2518,\n521,12343,\n6765,4558,\n525,7496,\n49407]]', font_size=18).move_to(Image_in_latent_Box).align_to(Image_in_latent_Box, LEFT+UP)
        In_latent=Group(Image_in_latent_Box, Image_in_latent_text)
        in_lanent_arrow_lable=Text('VAE Encoder', font_size=18).next_to(Image_in_latent_arrow, UP)
        from_latent_arrow=Arrow().next_to(In_latent, RIGHT )
        from_lanent_arrow_lable=Text('VAE Decoder', font_size=18).next_to(from_latent_arrow, UP)
        FromLatent_image=ImageMobject('0000034-1045145337-Austronaut riding horse on mars.png').next_to(from_latent_arrow, RIGHT)
        Size_text_1=Text('[1, 3, 512, 512]').next_to(Gen_Image, DOWN)
        Size_text_2=Text('[1, 3, 512, 512]').next_to(FromLatent_image, DOWN)
        Size_text_3=Text('[8, 1, 64, 64]').next_to(Image_in_latent_Box, DOWN)
        self.play(description.animate.become(description2), FadeOut(Vae_image), FadeIn(Size_text_1, Size_text_2, Size_text_3, Gen_Image, In_latent, Image_in_latent_arrow, in_lanent_arrow_lable, from_latent_arrow, from_lanent_arrow_lable, FromLatent_image))
        self.wait(3)
        description2=Text('Это позволяет уменьшить скрытую размерность с которой работает\nмодель, что сильно увеличевает её производительность\nкак при обучении, так и при генерации', font_size=32).move_to(description).align_to(description, LEFT+UP)
        self.play(description.animate.become(description2))
        self.wait(3)
        self.play(FadeOut( Size_text_1, Size_text_2, Size_text_3, Gen_Image, In_latent, Image_in_latent_arrow, in_lanent_arrow_lable, from_latent_arrow, from_lanent_arrow_lable, FromLatent_image))
        self.wait(1)
        title_1=Text('VAE encoder в Stable diffusion', font_size=36).to_corner(UP+LEFT)
        self.play(Write(title_1))
        description2=Text('Расмотрим детальнее энкодер VAE в Stable diffusion', font_size=32).move_to(description).align_to(description, LEFT+UP)
        self.play(description.animate.become(description2))
        self.wait(1)

                # Создаем объекты Text для каждого блока
        blocks = VGroup(
            Text("Вход"),
            Text("[conv_in]"),
            Text("[down_blocks]"),
            Text("[mid_block]"),
            Text("[up_blocks]"),
            Text("[conv_out]"),
            Text("Выход")
        )
        blocks.scale(0.5)
        blocks.shift(UP+LEFT)

                # Создаем описания для каждого блока
        descriptions = VGroup(
        Text("Исходные данные").scale(0.5),
        MathTex(r"\text{Свёртка: } f(x) = \text{ReLU}(Wx + b)").scale(0.5),
        Text("Понижение размерности").scale(0.5),
        MathTex(r"\text{Латентное пространство: } z = \frac{x - \mu}{\sigma}").scale(0.5),
        Text("Повышение размерности").scale(0.5),
        MathTex(r"\text{Восстановление: } \hat{x} = \text{Sigmoid}(Wz + b)").scale(0.5),
        Text("Реконструированные данные").scale(0.5)
        )

        # Располагаем блоки вертикально
        blocks.arrange(DOWN, center=False)

        # Создаем стрелки между блоками
        arrows = VGroup(*[
            Arrow(blocks[i].get_bottom(), blocks[i+1].get_top())
            for i in range(len(blocks) - 1)
        ])

         # Располагаем описания справа от блоков
        for desc, block in zip(descriptions, blocks):
            desc.next_to(block, RIGHT)

        # Добавляем все на сцену
        self.play(Write(blocks), Write(arrows), Write(descriptions))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class VAEdecoder(Scene):
    def construct(self):
        title_1=Text('VAE decoder', font_size=36).to_corner(UP+LEFT)
        self.play(Write(title_1))
        blocks = VGroup(
            Text("Вход"),
            Text("[conv_in]"),
            Text("[up_blocks]"),
            Text("[mid_block]"),
            Text("[conv_norm_out]"),
            Text("[conv_out]"),
            Text("Выход")
        )
        blocks.scale(0.5)
        blocks.arrange(DOWN, center=True, buff=0.5)
        blocks.to_edge(LEFT, buff=0.5)

        # Создаем описания для каждого блока
        descriptions = VGroup(
            Text("Исходные данные").scale(0.5),
            MathTex(r"\text{Свёртка: } f(x) = \text{ReLU}(Wx + b)").scale(0.5),
            Text("Увеличение размерности").scale(0.5),
            Text("Блок внимания и ResNet").scale(0.5),
            Text("Нормализация выхода").scale(0.5),
            MathTex(r"\text{Восстановление: } \hat{x} = \text{Sigmoid}(Wz + b)").scale(0.5),
            Text("Реконструированные данные").scale(0.5)
        )
        descriptions.arrange(DOWN, center=True, buff=0.5)
        descriptions.next_to(blocks, RIGHT, buff=2)

        # Создаем стрелки между блоками
        arrows = VGroup(*[
            Arrow(blocks[i].get_bottom(), blocks[i+1].get_top(), buff=0.1)
            for i in range(len(blocks) - 1)
        ])

        # Группируем блоки, описания и стрелки
        decoder = VGroup(blocks, descriptions, arrows)

        # Анимируем появление блоков, описаний и стрелок
        self.play(LaggedStart(*[Write(block) for block in blocks], lag_ratio=0.2))
        self.play(LaggedStart(*[Write(desc) for desc in descriptions], lag_ratio=0.2))
        self.play(Create(arrows))  # Используем Create вместо ShowCreation
        self.wait(2)

        # Анимируем выделение каждого блока и его описания
        for block, desc in zip(blocks, descriptions):
            self.play(Indicate(block), Indicate(desc))
            #self.wait(0.2)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class Unet(Scene):
    def construct(self):
        title_1=Text('Unet', font_size=36).to_corner(UP+LEFT)
        self.play(Write(title_1))
        self.wait(1)
        description=Text('Unet это главная часть Stable difusion использующая выход текстовой модели\nдля создания латентного изображения', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        self.play(Write(description))
        self.wait(2)
                # Создаем блоки U-Net

        # Создаем центральный блок (mid_block)
        mid_block = Rectangle(width=3, height=1, color=RED).set_fill(RED, opacity=0.5)
        mid_block.move_to(2*DOWN)

        # Создаем блоки кодировщика (encoder_blocks) с увеличивающимся расстоянием от mid_block
        encoder_blocks = VGroup()
        current_shift = 0  # Начальное смещение для первого блока
        for i in range(3):
            block = Rectangle(width=2-0.7*i, height=1+i*0.1, color=BLUE).set_fill(BLUE, opacity=0.5)
            block.next_to(mid_block, LEFT, buff=1 + i*1)
            # Устанавливаем вертикальное положение каждого блока, учитывая его высоту
            current_shift += 1 + i*0.1  # Увеличиваем смещение на высоту предыдущего блока
            block.shift(UP*current_shift)
            encoder_blocks.add(block)

        decoder_blocks = VGroup()
        current_shift = 0  # Начальное смещение для первого блока
        for i in range(3):
            block = Rectangle(width=2-0.7*i, height=1+i*0.1, color=GREEN).set_fill(GREEN, opacity=0.5)
            block.next_to(mid_block, RIGHT, buff=1 + i*1)
            # Устанавливаем вертикальное положение каждого блока, учитывая его высоту
            current_shift += 1 + i*0.1  # Увеличиваем смещение на высоту предыдущего блока
            block.shift(UP*current_shift)
            decoder_blocks.add(block)
        
        skip_connections = VGroup()
        for i in range(3):
            skip_connection = Arrow(
                encoder_blocks[i].get_edge_center(RIGHT),
                decoder_blocks[i].get_edge_center(LEFT),
                buff=0.1,
                color=PURPLE
            )
            skip_connections.add(skip_connection)
        
        # Добавляем блоки на сцену
        self.play(Create(mid_block), Create(encoder_blocks), Create(decoder_blocks), Create(skip_connections))
        self.wait(2)

        input_arrows = VGroup()
        for i in range(3):
            input_arrow_encoder = Arrow(
                encoder_blocks[i].get_top() + UP*0.5,  # начальная точка немного выше блока
                encoder_blocks[i].get_top(),  # конечная точка на верхней границе блока
                buff=0.1,
                color=WHITE,
                max_tip_length_to_length_ratio=0.2,  # Увеличиваем размер кончика стрелки
                max_stroke_width_to_length_ratio=5,  # Увеличиваем толщину стрелки
                stroke_width=6  # Устанавливаем толщину линии стрелки
            )
            input_arrow_decoder = Arrow(
                decoder_blocks[i].get_top() + UP*0.5,  # начальная точка немного выше блока
                decoder_blocks[i].get_top(),  # конечная точка на верхней границе блока
                buff=0.1,
                color=WHITE,
                max_tip_length_to_length_ratio=0.2,  # Увеличиваем размер кончика стрелки
                max_stroke_width_to_length_ratio=5,  # Увеличиваем толщину стрелки
                stroke_width=6  # Устанавливаем толщину линии стрелки
            )
            mid_arrow= Arrow(mid_block.get_top()+ UP*0.5, mid_block.get_top(), buff=0.1, color=WHITE, max_tip_length_to_length_ratio=0.2,  # Увеличиваем размер кончика стрелки
                max_stroke_width_to_length_ratio=5,  # Увеличиваем толщину стрелки
                stroke_width=6  )
                # Устанавливаем толщину линии стрелки)
            input_arrows.add(input_arrow_encoder, input_arrow_decoder, mid_arrow)
        
        description2=Text('Обычно Unet используется для анализа изображений в задачах\nсегментации и состоит сверточных блоков соендиненых skip connections', font_size=28).move_to(description).align_to(description, LEFT+UP)
        self.play(description.animate.become(description2))
        self.wait(1)


        #start_index = 'Обычно Unet используется для анализа изображений в задачах\nсегментации и состоит сверточных блоков соендиненых skip connections'.find("skip connections")
        #end_index = start_index + len("skip connections")
        phrase_to_circumscribe = description[len("skip connections")*-1:-1]


        self.play(Circumscribe(phrase_to_circumscribe, color=BLUE, fade_out=True), *[Indicate(skip_connection, color=YELLOW) for skip_connection in skip_connections] )
        #self.play()

        self.wait(2)
        description.shift(UP)
        description2=Text('В Stable diffusion в Unet присутвует вход для данных полученых из CLIP\nчто позволяет изменять изрображение в соответсвии с описанием', font_size=28).move_to(description).align_to(description, LEFT+UP)
        
        
        self.play(Create(input_arrows), FadeOut(title_1), description.animate.become(description2))
        self.play(*[Indicate(input_arrows, color=YELLOW) for input_arrows in input_arrows], description.animate.become(description2))
        self.wait(2)
###############################################################################################################################################
        description2=Text('Unet состоит из энкодера, декодера, и bottleneck', font_size=28).move_to(description).align_to(description, LEFT+UP)
        self.play( description.animate.become(description2))
        self.wait(1)
        start_index = description2.text.find('энкодера')
        end_index = start_index + len('энкодера')

        phrase_to_circumscribe = description2[start_index:end_index]

        #phrase_to_circumscribe = description[description.text.index('энкодера'):description.text.index('энкодера')+len('энкодера')]
        self.play(Indicate(phrase_to_circumscribe, color=YELLOW, fade_out=True), *[Indicate(encoder_blocks, color=YELLOW) for encoder_blocks in encoder_blocks] )
        self.play(FadeOut(phrase_to_circumscribe))
        self.wait(1)
        start_index = description2.text.find('декодера')
        end_index = start_index + len('декодера')
        phrase_to_circumscribe = description2[start_index:end_index]
        self.play(Indicate(phrase_to_circumscribe, fade_out=True, color=YELLOW), *[Indicate(decoder_blocks, color=YELLOW) for decoder_blocks in decoder_blocks] )
        self.play(FadeOut(phrase_to_circumscribe))

        self.wait(1)
        start_index = description2.text.find('bottleneck')
        end_index = start_index + len('bottleneck')
        phrase_to_circumscribe = description2[start_index:end_index]
        self.play(Indicate(phrase_to_circumscribe, color=YELLOW, fade_out=True), Indicate(mid_block, color=YELLOW) )
        self.play(FadeOut(phrase_to_circumscribe))


        self.wait(2)
        description2=Text('Для передачи используется блок Cross attention', font_size=28).move_to(description).align_to(description, LEFT+UP)
        self.play( FadeOut(description2,  decoder_blocks, encoder_blocks, input_arrows, skip_connections), description.animate.become(description2), mid_block.animate.scale(3))

        arrow_left_1 = Arrow(mid_block.get_left()+LEFT*10, mid_block.get_left()+LEFT*0.5 )
        arrow_left_2 = Arrow(mid_block.get_left()+LEFT*10+DOWN*0.5, mid_block.get_left()+LEFT*0.5+ DOWN*0.5)
        arrow_top = Arrow(UP, mid_block.get_top())
         # Добавляем стрелки на сцену
        self.play(GrowArrow(arrow_left_1), GrowArrow(arrow_left_2), GrowArrow(arrow_top))
        self.wait(2)
        arrow_descr1=Text('Q').next_to(arrow_top, RIGHT)
        arrow_descr2=Text('K').move_to(mid_block.get_left()+LEFT*0.5+UP)
        arrow_descr3=Text('V').move_to(mid_block.get_left()+LEFT*0.5+ DOWN*0.5+DOWN)
        crossatantion_image=ImageMobject('https-__qiita-image-store.s3.amazonaws.com_0_123589_fce9f6ab-6b88-01c6-b8f0-fd45b9193d3e.png'). scale(0.5).next_to(arrow_descr1, RIGHT).shift(UP*1.5+RIGHT)


        description2=Text('Что подается в Cross attention', font_size=28).move_to(description).align_to(description, LEFT+UP)
        self.play(description.animate.become(description2), FadeIn(arrow_descr1, arrow_descr2, arrow_descr3, crossatantion_image))
        self.wait(2)
        description2=Text('В данном случае:\nQ-текстовое описание\nK-латентное изображение\nV-веса полученые при обучении модели', font_size=28).move_to(description).align_to(description, LEFT+UP)
        self.play(description.animate.become(description2))
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

class Diffusion(Scene):
    def construct(self):
        title_1=Text('Процесс дифузии', font_size=36).to_corner(UP+LEFT)
        self.play(FadeIn(title_1))

        image_folder = "intermediate"  # Замените на путь к вашей папке с изображениями
        images = os.listdir(image_folder)
        prev_image = None
        for image_file in images:
            if prev_image is not None:
                self.remove(prev_image)
            image_path = os.path.join(image_folder, image_file)
            image = ImageMobject(image_path)
            self.add(image)
            self.wait(0.2)
            prev_image = image

        self.play(FadeOut(prev_image))
        self.wait(0.5)
        description=Text('Теперь когда мы разобрали каждую часть Stable diffusion возможно разобрать\nпроцесс создания изображения', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        self.play(Write(description))
        self.wait(1.5)
        self.play(FadeOut(description))
        description2=Text('Все начинаеться с латентного изображения состоящего из шума', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        noise_image=ImageMobject('noise.png')
        self.play(FadeIn(description2, noise_image))
        self.wait(2)
        description=Text('Данный шум передается в Unet', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        self.play(FadeOut(description2, noise_image), FadeIn(description))


        mid_block = Rectangle(width=3, height=1, color=RED).set_fill(RED, opacity=0.5)
        mid_block.move_to(2*DOWN)

        # Создаем блоки кодировщика (encoder_blocks) с увеличивающимся расстоянием от mid_block
        encoder_blocks = VGroup()
        current_shift = 0  # Начальное смещение для первого блока
        for i in range(3):
            block = Rectangle(width=2-0.7*i, height=1+i*0.1, color=BLUE).set_fill(BLUE, opacity=0.5)
            block.next_to(mid_block, LEFT, buff=1 + i*1)
            # Устанавливаем вертикальное положение каждого блока, учитывая его высоту
            current_shift += 1 + i*0.1  # Увеличиваем смещение на высоту предыдущего блока
            block.shift(UP*current_shift)
            encoder_blocks.add(block)

        decoder_blocks = VGroup()
        current_shift = 0  # Начальное смещение для первого блока
        for i in range(3):
            block = Rectangle(width=2-0.7*i, height=1+i*0.1, color=GREEN).set_fill(GREEN, opacity=0.5)
            block.next_to(mid_block, RIGHT, buff=1 + i*1)
            # Устанавливаем вертикальное положение каждого блока, учитывая его высоту
            current_shift += 1 + i*0.1  # Увеличиваем смещение на высоту предыдущего блока
            block.shift(UP*current_shift)
            decoder_blocks.add(block)
        
        skip_connections = VGroup()
        for i in range(3):
            skip_connection = Arrow(
                encoder_blocks[i].get_edge_center(RIGHT),
                decoder_blocks[i].get_edge_center(LEFT),
                buff=0.1,
                color=PURPLE
            )
            skip_connections.add(skip_connection)
        
        # Добавляем блоки на сцену
        self.play(Create(mid_block), Create(encoder_blocks), Create(decoder_blocks), Create(skip_connections))

        self.wait(1)

        Unet= Group(mid_block, encoder_blocks, decoder_blocks,skip_connections )

        self.play(Unet.animate.scale(0.5).shift(RIGHT*4))
        self.wait(1)
        description2=Text('Текстовое описание изображения передается в модель CLIP', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        PromtBox=Square().shift(LEFT*6)
        PromtText=Text('Austronaut riding\nhorse on mars', font_size=18).move_to(PromtBox).align_to(PromtBox, LEFT+UP)
        Promt= Group(PromtText, PromtBox)
        Clip_square= Rectangle(2, 1, ).scale(0.7).set_fill(RED, opacity=0.5).shift(LEFT*2)
        Clip_text=Text('CLIP').move_to(Clip_square)
        Clip=Group(Clip_square, Clip_text)

        # Создаем стрелку от Promt к Clip
        arrow_promt_to_clip = Arrow(Promt.get_right(), Clip.get_left(), buff=0.1)
        
        # Создаем стрелку от Clip к Unet
        arrow_clip_to_unet = Arrow(Clip.get_right(), Unet.get_left(), buff=0.1)

        # Позиционируем noise_image
        noise_image.scale(0.5).next_to(Clip, DOWN, buff=1.5)

        # Создаем стрелку от noise_image к Unet
        arrow_noise_to_unet = Arrow(noise_image.get_right(), Unet.get_left(), buff=0.1)

        # Анимируем появление объектов и стрелок
        self.play(FadeIn(description2, arrow_promt_to_clip, arrow_clip_to_unet, arrow_noise_to_unet, noise_image, Promt, Clip), FadeOut(description))
        self.wait(2)
        description=Text('После чего Unet предсказывает шумоподавление, и в зависимости от семплера\nвычетает данный шум из латентного изображения', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        
        self.play(FadeOut(description2,arrow_promt_to_clip, arrow_clip_to_unet, arrow_noise_to_unet, noise_image, Promt, Clip ), FadeIn(description), Unet.animate.to_edge(LEFT))
        self.wait(1)

        SecondImage= ImageMobject(r'intermediate\0000000-1045145337-Austronaut riding horse on mars.png').scale(0.5).to_edge(RIGHT).shift(UP)
        noise_image.next_to(SecondImage, DOWN, buff=2)

        

        arrow_to_second_image= Arrow(noise_image.get_top(), SecondImage.get_bottom(), buff=0.1)
        sampler_text=Text('Семплер', font_size=18).next_to(arrow_to_second_image, LEFT, buff=2)
        arrow_to_sampler=Arrow(Unet.get_right(), arrow_to_second_image.get_left())
        text_for_predicted_noise= Text('Предсказаный шум', font_size=18).next_to(arrow_to_sampler, UP)
        self.play(FadeIn(SecondImage, noise_image, arrow_to_second_image, sampler_text, arrow_to_sampler, text_for_predicted_noise ))#,  sampler_text, arrow_to_second_image, arrow_to_sampler, text_for_predicted_noise))
        self.wait(2)
        description2=Text('Так завершается шаг, после чего новое латентное изображение подается на вход Unet\nпроцесс повторяется столько шагов сколько было установлено', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        
        arrowTO_unet=Arrow(SecondImage.get_left(), Unet.get_right(), buff=0.1)

        self.play(FadeIn(arrowTO_unet, description2), FadeOut(description) )
        self.wait(2)



        for image_file in images:
            if SecondImage is not None:
                self.remove(SecondImage)
            image_path = os.path.join(image_folder, image_file)
            
            self.remove(noise_image)
            noise_image = ImageMobject(image_path).scale(0.5).next_to(arrow_to_second_image, UP, buff=0.1)
            
            self.add(noise_image)
            SecondImage=ImageMobject(image_path).scale(0.5).next_to(arrow_to_second_image, DOWN, buff=0.1)

            
            self.add(SecondImage)
            self.wait(0.2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)



class VaeEnd(Scene):
    def construct(self):
        title_1=Text('Получение финального изображения', font_size=36).to_corner(UP+LEFT)
        self.play(FadeIn(title_1))
        description=Text('После всего выполнения необходимого количества шагов необходимо\nперевести латентное изображение в пиксельное с помощью VAE', font_size=25).next_to(title_1, DOWN).align_to(title_1, LEFT)
        self.play(FadeIn(description))
        self.wait(2)

        LatentImagebox=Square().to_edge(LEFT)
        LatentText=Text('[8, 1, 64, 64]', font_size=18).move_to(LatentImagebox).align_to(LatentImagebox, LEFT+UP)
        Latent=Group(LatentImagebox, LatentText)

        
        VAEbox=Square().set_fill(RED, opacity=0.5)
        VAEText=Text('VAE', font_size=18).move_to(VAEbox)
        VAE=Group(VAEbox, VAEText)
        arrow_to_vae=Arrow(Latent.get_right(),VAE.get_left(), buff=0.1)

        FinalImage=ImageMobject(r'0000034-1045145337-Austronaut riding horse on mars.png').to_edge(RIGHT)

        arrow_to_image=Arrow(VAE.get_right(), FinalImage.get_left(), buff=0.1)


        self.play(FadeIn(Latent, VAE, arrow_to_vae, FinalImage, arrow_to_vae, arrow_to_image))

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        Final=Text('Спасибо за внимание')
        self.play(FadeIn(Final))

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)




class MainScene(Scene):
     def construct(self):
        TextToImageArchitecture.construct(self)
        ModelContent.construct(self)
        ClipFor15.construct(self)
        VAE.construct(self)
        VAEdecoder.construct(self)
        Diffusion.construct(self)
        VaeEnd.construct(self)



