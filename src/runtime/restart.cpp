#include "fuse/runtime/restart.hpp"

namespace fuse {

namespace {
constexpr char kTimeName[] = "__point/time";
constexpr char kPeriodName[] = "__point/period_index";
}  // namespace

Result<void> write_restart_point(Runner& runner, ComponentSet& components,
                                 RestartWriter& writer, const Rational& time,
                                 std::int64_t period_index) {
  Result<void> r = writer.put_scalar_rational(kTimeName, time);
  if (!r.ok()) return r;
  r = writer.put_scalar(kPeriodName, static_cast<double>(period_index));
  if (!r.ok()) return r;
  r = runner.write_restart_fields(writer, components);
  if (!r.ok()) return r;
  return writer.finish_point();
}

Result<void> check_restart_rank_count(const ComponentSet& components,
                                      std::uint32_t written_by_rank_count,
                                      std::uint32_t target_rank_count) {
  if (written_by_rank_count == target_rank_count) return Result<void>::success();
  for (const Component& c : components) {
    if (c.supports_restart_form_b() && !c.supports_restart_form_a()) {
      return Failure{"E-RESTART-FORMB-ONLY-RANK-CHANGE",
                     "check_restart_rank_count: component '" + c.name() +
                         "' offers only Form B (same-decomposition-only), "
                         "and this restart point was written at " +
                         std::to_string(written_by_rank_count) +
                         " ranks but is being restored at " +
                         std::to_string(target_rank_count)};
    }
  }
  return Result<void>::success();
}

Result<void> restore_restart_point(Runner& runner, ComponentSet& components,
                                   RestartReader& reader,
                                   bool allow_accelerator_reset_on_mismatch,
                                   bool& any_accelerator_reset,
                                   Rational& out_time,
                                   std::int64_t& out_period_index) {
  const Result<Rational> time = reader.get_scalar_rational(kTimeName);
  if (!time.ok()) return time.error();
  const Result<double> period = reader.get_scalar(kPeriodName);
  if (!period.ok()) return period.error();
  out_time = time.value();
  out_period_index = static_cast<std::int64_t>(period.value());
  const Result<void> r = runner.restore_restart_fields(
      reader, components, allow_accelerator_reset_on_mismatch,
      any_accelerator_reset);
  if (!r.ok()) return r;
  // The Runner's own clock is not a fuse::Component and restore_restart_
  // fields() does not touch it -- see Runner::set_current_period_index()'s
  // own doc comment for why this call belongs here, after fields succeed,
  // rather than folded into restore_restart_fields() itself.
  runner.set_current_period_index(out_period_index);
  return Result<void>::success();
}

}  // namespace fuse
