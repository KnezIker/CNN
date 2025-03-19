#include <stdio.h>
#include <stdint.h>

// Deklaracije za inline asembler
void cmul(int32_t ra, int32_t rb);  // cmul ra, rb
int32_t cget(void);                 // cget rd
void crst(void);                    // crst

int main() {
    int32_t result;

    // Resetuj akumulator
    crst();

    // Testiranje cmul i cget
    int32_t a = 0x00020000;  // 2.0 u Q15.16
    int32_t b = 0x00030000;  // 3.0 u Q15.16

    cmul(a, b);  // 2.0 * 3.0 = 6.0
    result = cget();
    printf("Rezultat nakon prvog množenja: %d\n", result);

    int32_t c = 0x00010000;  // 1.0 u Q15.16
    int32_t d = 0x00040000;  // 4.0 u Q15.16

    cmul(c, d);  // 1.0 * 4.0 = 4.0
    result = cget();
    printf("Rezultat nakon drugog množenja: %f\n", result);  // Očekivano: 10.0 (6.0 + 4.0)

    // Resetuj akumulator i testiraj ponovo
    crst();

    int32_t e = 0x00050000;  // 5.0 u Q15.16
    int32_t f = 0x00020000;  // 2.0 u Q15.16

    cmul(e, f);  // 5.0 * 2.0 = 10.0
    result = cget();
    printf("Rezultat nakon resetovanja i trećeg množenja: %f\n", result);  // Očekivano: 10.0

    return 0;
}

// Implementacije asemblerskih instrukcija
void cmul(int32_t ra, int32_t rb) {
__asm__ volatile ("cmul %0, %1" : "r"(ra) : "r"(rb));
}

int32_t cget(void) {
    int32_t result;
    __asm__ volatile ("cget %0" : "=r" (result));
    return result;
}

void crst(void) {
    __asm__ volatile ("crst");
}




