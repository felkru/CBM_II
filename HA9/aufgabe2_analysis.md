
## Analyse der adaptiven Schrittweite (Aufgabe 2d und 2e)

**d) Was beobachten Sie im Vergleich zu den anderen Algorithmen?**

Der adaptive Euler-Algorithmus mit Halbierung der Schrittweite (Half-Step Method) unterscheidet sich von festen Schrittweiten-Algorithmen (wie dem einfachen Euler-Verfahren) und anderen adaptiven Verfahren (wie Runge-Kutta-Methoden höherer Ordnung) hauptsächlich in seiner Fähigkeit, die Schrittweite dynamisch an die lokale Dynamik des Systems anzupassen. 

Im Vergleich zu einem **festen Schrittweiten-Euler-Verfahren** bietet der adaptive Ansatz folgende Vorteile:

*   **Effizienz:** In Bereichen, in denen sich die Lösung langsam ändert (z.B. wenn die Funktion flach ist), kann der adaptive Algorithmus größere Schrittweiten wählen, was Rechenzeit spart. Bei festen Schrittweiten müsste man für das gesamte Intervall eine kleine Schrittweite wählen, um die Genauigkeit in kritischen Bereichen zu gewährleisten, was zu unnötig vielen Berechnungen in unkritischen Bereichen führt.
*   **Genauigkeit:** In Bereichen, in denen sich die Lösung schnell ändert (z.B. bei starken Krümmungen oder Oszillationen), reduziert der adaptive Algorithmus die Schrittweite. Dies gewährleistet, dass die lokale Fehlertoleranz eingehalten wird und die numerische Lösung die wahre Lösung genauer approximiert. Ein fester Schrittweiten-Algorithmus würde hier entweder ungenau werden oder müsste von vornherein eine sehr kleine Schrittweite verwenden, was ineffizient wäre.
*   **Stabilität:** Bei manchen Systemen kann eine zu große Schrittweite zu numerischer Instabilität führen. Der adaptive Ansatz hilft, dies zu vermeiden, indem er die Schrittweite bei Bedarf verkleinert.

Im Vergleich zu **Runge-Kutta-Methoden höherer Ordnung** (die oft auch adaptive Schrittweitenkontrolle implementieren) ist der hier implementierte adaptive Euler-Algorithmus (Half-Step Method) konzeptionell einfacher, aber auch weniger effizient und genauer. Runge-Kutta-Methoden verwenden in der Regel mehrere Funktionsauswertungen pro Schritt, um eine höhere Genauigkeitsordnung zu erreichen, und nutzen oft komplexere Fehlerschätzer. Die Half-Step Method ist eine grundlegende Form der adaptiven Schrittweitenkontrolle, die hauptsächlich dazu dient, das Konzept der Fehlerkontrolle zu demonstrieren.

**e) Wie verhält sich die adaptive Schrittweite?**

Die adaptive Schrittweite verhält sich in Abhängigkeit von der lokalen Fehlertoleranz (epsilon) und der Dynamik der zu lösenden Differentialgleichung. Basierend auf der Implementierung und dem typischen Verhalten solcher Algorithmen beobachten wir:

*   **Anpassung an die Steigung/Krümmung der Lösung:** Wenn die Lösung der Differentialgleichung sich stark ändert (d.h., die Ableitung groß ist oder die Funktion stark gekrümmt ist), wird der lokale Fehler `e` tendenziell größer sein. Um die Fehlertoleranz `epsilon` einzuhalten, wird der Algorithmus die Schrittweite `h` verkleinern. Umgekehrt, wenn sich die Lösung langsam ändert, ist der lokale Fehler `e` kleiner, und der Algorithmus kann die Schrittweite `h` vergrößern.
*   **Oszillationen der Schrittweite:** Es ist typisch, dass die adaptive Schrittweite nicht monoton verläuft, sondern oszilliert. Der Algorithmus versucht, die größte mögliche Schrittweite zu finden, die die Fehlertoleranz einhält. Wenn er eine zu große Schrittweite wählt, wird der Fehler zu groß, und er muss die Schrittweite im nächsten Schritt reduzieren. Wenn er eine zu kleine Schrittweite wählt, ist der Fehler sehr klein, und er kann die Schrittweite im nächsten Schritt vergrößern. Dies führt zu einem ständigen Anpassen und Suchen nach der optimalen Schrittweite.
*   **Einfluss von `epsilon`:** Eine kleinere Fehlertoleranz `epsilon` führt dazu, dass der Algorithmus kleinere Schrittweiten wählt, um eine höhere Genauigkeit zu erreichen. Eine größere `epsilon` erlaubt größere Schrittweiten, was die Berechnung beschleunigt, aber die Genauigkeit reduziert.
*   **Einfluss von `hmax`:** Die maximale Schrittweite `hmax` (hier mit Default-Wert 1) begrenzt, wie groß die Schrittweite werden kann. Dies ist wichtig, um zu verhindern, dass der Algorithmus in sehr unkritischen Bereichen extrem große Schritte macht, die möglicherweise andere Probleme verursachen könnten.

Im gezeigten Plot der adaptiven Schrittweite (adaptive_euler_results.png, unterer Plot) ist dieses oszillierende Verhalten deutlich zu erkennen. Die Schrittweite passt sich dynamisch an, um die Balance zwischen Genauigkeit und Effizienz zu finden. In diesem vereinfachten Beispiel, wo die Funktion `y` exponentiell abnimmt, könnte man erwarten, dass die Schrittweite tendenziell größer wird, wenn `y` kleiner wird und sich langsamer ändert, oder dass sie sich an die lokale Krümmung anpasst, um den Fehler konstant zu halten.


