import { LinearColorMapper } from "models/mappers/linear_color_mapper"
import type { LinearScanData } from "models/mappers/linear_color_mapper"
import { clamp } from "core/util/math"
import type * as p from "core/properties"

export namespace GammaColorMapper {
    export type Attrs = p.AttrsOf<Props>

    export type Props = LinearColorMapper.Props & {
        gamma: p.Property<number>
    }
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

    override index_to_value(index: number): number {
        const scan_data = this._scan_data as LinearScanData
        return scan_data.min + scan_data.normed_interval * index / scan_data.norm_factor
    }

    override value_to_index(value: number, palette_length: number): number {
        const scan_data = this._scan_data as LinearScanData

        // This handles the edge case where value == high, since the code below maps
        // values exactly equal to high to palette.length when it should be one less.
        if (value == scan_data.max)
            return palette_length - 1

        // const normed_value = (value - scan_data.min) * scan_data.norm_factor
        // const index = Math.floor(normed_value / scan_data.normed_interval)
        const fraction = Math.pow((value - scan_data.min) / (scan_data.max - scan_data.min), this.gamma)
        const index = Math.floor(fraction * palette_length)
        // console.log(fraction, index, palette_length)
        return clamp(index, -1, palette_length)
    }

    static {
        this.define<GammaColorMapper.Props>(({ Number }) => ({
            gamma: [Number, 1.0],
        }))
    }
}