#include "contracts.h"
#include <cassert>
#include <iostream>
#include <set>

using namespace probe_contracts;
int main() {
    Price p{{100, 100, 0, 0, 0, 0, 0}, 256, 32, 16, 16};
    Domains d{4096, 64, 2048, 128, 1024, 64};
    assert(!admit(d, p)); // Shared Metal is larger than CPU availability and still fits.
    auto bad = d; bad.cpu_available = 63; assert(admit(bad, p));
    bad = d; bad.mwss = 383; assert(admit(bad, p));
    bad = d; bad.physical = 447; assert(admit(bad, p));
    bad = d; bad.max_buffer = 99; assert(admit(bad, p));
    for (auto member : {&Domains::physical, &Domains::mwss, &Domains::max_buffer, &Domains::page}) {
        bad = d; bad.*member = 0; assert(admit(bad, p));
    }
    bad = d; bad.page = 3; assert(admit(bad, p));
    bad = d; bad.cpu_available = 4097; assert(admit(bad, p));
    assert(pages(1, 16384) == 16384 && pages(16384, 16384) == 16384);
    assert(pages(16385, 16384) == 32768 && pages(0, 16384) == 0);
    bool overflow = false;
    try { pages(UINT64_MAX, 16384); } catch (const std::runtime_error &) { overflow = true; }
    assert(overflow);
    for (int count = 1; count <= 64; ++count) for (bool invalid : {false, true}) {
        std::set<int> all;
        const int limit = 8 * (count + 2) + 2;
        int invalid_count = 0;
        for (int expert = 0; expert < 8; ++expert) for (int i = 0; i < count; ++i) {
            const int slot = fixture_slot(expert, i, count, false, invalid, limit);
            if (slot < 0 || slot >= limit) {
                assert(invalid && ((i == 3 && slot == -1) || (i == 7 && slot == limit)));
                ++invalid_count;
            } else {
                assert(slot >= expert * (count + 2) && slot < expert * (count + 2) + count);
                assert(all.insert(slot).second);
            }
            assert(fixture_slot(expert, i, count, true, invalid, limit) == i);
        }
        const int expected_invalid = invalid ? 8 * ((count > 3) + (count > 7)) : 0;
        assert(invalid_count == expected_invalid && int(all.size()) == 8 * count - expected_invalid);
        if (std::gcd(17, count) == 1)
            for (int i = 0; i < count; ++i) assert(fixture_slot(0, i, count, false, false, limit) == i * 17 % count);
    }
    for (int count : {17, 34, 51}) {
        std::set<int> legacy;
        for (int i = 0; i < count; ++i) legacy.insert(i * 17 % count);
        assert(int(legacy.size()) != count);
    }
    std::array<std::array<uint64_t, 4>, 32> transpose{};
    for (int lane = 0; lane < 32; ++lane) {
        const int fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
        const int quarter = ((lane >> 2) & 2) | (lane & 1);
        for (int r = 0; r < 4; ++r) transpose[lane][r] = (fm + 8 * quarter) * 4 + r;
    }
    for (int bit : {1, 2}) for (int r = 0; r < 4; ++r) if (!(r & bit)) {
        const auto before = transpose;
        for (int lane = 0; lane < 32; ++lane) {
            const int quarter = ((lane >> 2) & 2) | (lane & 1);
            const int partner = lane ^ (bit == 1 ? 1 : 8);
            const int receive_register = r + ((quarter & bit) ? 0 : bit);
            const int send_register = r + ((quarter & bit) ? bit : 0);
            transpose[lane][receive_register] = before[partner][send_register];
        }
    }
    for (int lane = 0; lane < 32; ++lane) for (int r = 0; r < 4; ++r) {
        const int fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
        const int quarter = ((lane >> 2) & 2) | (lane & 1);
        assert(transpose[lane][r] == uint64_t((fm + 8 * r) * 4 + quarter));
    }
    std::cout << "domain, rounding, overflow, all-count slots and two-stage transpose properties pass\n";
}
