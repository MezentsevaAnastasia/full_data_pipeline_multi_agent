# Annotation Specification: sentiment_classification

## Задача

- **Тип задачи:** sentiment_classification
- **Модальность:** text
- **Количество классов:** 2
- **Объём данных:** 100 примеров

---

## Классы

### negative

**Определение:** Текст выражает отрицательное мнение, критику, неудовольствие или разочарование.

**Примеры:**

1. "a miniscule little bleep on the film radar , but one that many more people should check out"
2. "the animation and backdrops are lush and inventive , yet return to neverland never manages to take us to that elusive , lovely place where we suspend our disbelief ."
3. "beware the quirky brit-com . they can and will turn on a dime from oddly humorous to tediously sentimental ."

### positive

**Определение:** Текст выражает положительное мнение, одобрение, удовлетворение или похвалу.

**Примеры:**

1. "an infectious cultural fable with a tasty balance of family drama and frenetic comedy ."
2. "" mr . deeds " is suitable summer entertainment that offers escapism without requiring a great deal of thought ."
3. "that 'alabama' manages to be pleasant in spite of its predictability and occasional slowness is due primarily to the perkiness of witherspoon ( who is always a joy to watch , even when her material is"

---

## Граничные случаи

1. **Сарказм / ирония:** текст формально положительный, но по смыслу негативный. Размечайте по *истинному* смыслу.
2. **Смешанное мнение:** текст содержит и плюсы, и минусы. Выберите доминирующий тон.
3. **Нейтральные факты:** текст описывает факты без оценки. Если нет класса 'neutral' — выберите ближайший.
4. **Пример с низкой уверенностью** (confidence=0.59): "there must be an audience that enjoys the friday series , but i wouldn't be interested in knowing any of them personally…"
5. **Пример с низкой уверенностью** (confidence=0.58): "it will grip even viewers who aren't interested in rap , as it cuts to the heart of american society in an unnerving way…"

---

## Инструкции для разметчика

1. Прочитайте текст полностью перед принятием решения.
2. Выберите **один** класс, наиболее подходящий к тексту.
3. Если текст неоднозначный — отметьте как требующий обсуждения.
4. Если текст не относится ни к одному классу — пропустите.
5. Обращайте внимание на сарказм и иронию.
