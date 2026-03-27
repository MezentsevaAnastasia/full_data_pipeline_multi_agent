# Annotation Specification: sentiment_classification

## Задача

- **Тип задачи:** sentiment_classification
- **Модальность:** text
- **Количество классов:** 2
- **Объём данных:** 1100 примеров

---

## Классы

### negative

**Определение:** Текст выражает отрицательное мнение, критику, неудовольствие или разочарование.

**Примеры:**

1. "director uwe boll and the actors provide scant reason to care in this crude '70s throwback ."
2. "a poorly scripted , preachy fable that forgets about unfolding a coherent , believable story in its zeal to spread propaganda ."
3. "'it's painful to watch witherspoon's talents wasting away inside unnecessary films like legally blonde and sweet home abomination , i mean , alabama . '"

### positive

**Определение:** Текст выражает положительное мнение, одобрение, удовлетворение или похвалу.

**Примеры:**

1. "the production design , score and choreography are simply intoxicating ."
2. "'frailty " starts out like a typical bible killer story , but it turns out to be significantly different ( and better ) than most films with this theme ."
3. "it proves quite compelling as an intense , brooding character study ."

---

## Граничные случаи

1. **Сарказм / ирония:** текст формально положительный, но по смыслу негативный. Размечайте по *истинному* смыслу.
2. **Смешанное мнение:** текст содержит и плюсы, и минусы. Выберите доминирующий тон.
3. **Нейтральные факты:** текст описывает факты без оценки. Если нет класса 'neutral' — выберите ближайший.
4. **Пример с низкой уверенностью** (confidence=0.52): "at times auto focus feels so distant you might as well be watching it through a telescope . yet in its own aloof , unrea…"
5. **Пример с низкой уверенностью** (confidence=0.57): "Logic will get you from A to Z; imagination will get you everywhere.…"

---

## Инструкции для разметчика

1. Прочитайте текст полностью перед принятием решения.
2. Выберите **один** класс, наиболее подходящий к тексту.
3. Если текст неоднозначный — отметьте как требующий обсуждения.
4. Если текст не относится ни к одному классу — пропустите.
5. Обращайте внимание на сарказм и иронию.
