import { LinearColorMapper } from "models/mappers/linear_color_mapper"
import type { LinearScanData } from "models/mappers/linear_color_mapper"
import { clamp } from "core/util/math"
import type { Arrayable } from "core/types"
import type * as p from "core/properties"

export namespace GammaColorMapper {
    export type Attrs = p.AttrsOf<Props>

    export type Props = LinearColorMapper.Props & {
        gamma: p.Property<number>
    }
}

export type GammaScanData = LinearScanData & {
    do_gamma: boolean
    gamma_t: number
}

export interface GammaColorMapper extends GammaColorMapper.Attrs { }

export class GammaColorMapper extends LinearColorMapper {
    // Generally, the ``__name__`` class attribute should match the name of
    // the corresponding Python class exactly. TypeScript matches the name
    // automatically during compilation, so, barring some special cases, you
    // don't have to do this manually. This helps avoid typos, which stop
    // serialization/deserialization of the model.
    static override __name__ = "GammaColorMapper"

    declare properties: GammaColorMapper.Props

    constructor(attrs?: Partial<GammaColorMapper.Attrs>) {
        super(attrs)
    }

    override connect_signals(): void {
        super.connect_signals()
        const { gamma } = this.properties
        this.on_change(gamma, () => this.update_data())
    }

    protected override scan(data: Arrayable<number>, n: number): GammaScanData {
        const lin_data = super.scan(data, n)
        const do_gamma = this.gamma != 0
        var gamma_t = 1 + Math.abs(this.gamma)
        if (this.gamma < 0.) {
            gamma_t = 1 / gamma_t
        }
        return { ...lin_data, do_gamma: do_gamma, gamma_t: gamma_t }
    }

    override index_to_value(index: number): number {
        const scan_data = this._scan_data as GammaScanData
        if (!scan_data.do_gamma) {
            return scan_data.min + ((scan_data.normed_interval * index) / scan_data.norm_factor)
        } else {
            return scan_data.min + (Math.pow(scan_data.normed_interval * index, 1 / scan_data.gamma_t) / scan_data.norm_factor)
        }
    }

    override value_to_index(value: number, palette_length: number): number {
        const scan_data = this._scan_data as GammaScanData

        // This handles the edge case where value == high, since the code below maps
        // values exactly equal to high to palette.length when it should be one less.
        if (value == scan_data.max)
            return palette_length - 1

        const normed_value = (value - scan_data.min) * scan_data.norm_factor

        if (normed_value < 0.) {
            return -1
        } else if (normed_value > 1.) {
            return palette_length
        }

        var fraction = normed_value
        if (scan_data.do_gamma) {
            fraction = Math.pow(normed_value, scan_data.gamma_t)
        }
        const index = Math.floor(fraction / scan_data.normed_interval)
        return clamp(index, -1, palette_length)
    }

    static {
        this.define<GammaColorMapper.Props>(({ Number }) => ({
            gamma: [Number, 0.0],
        }))
    }
}
