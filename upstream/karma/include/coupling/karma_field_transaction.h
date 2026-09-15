#ifndef KARMA_COUPLING_FIELD_TRANSACTION_H
#define KARMA_COUPLING_FIELD_TRANSACTION_H

#include "karma_collective_outcome.h"
#include "karma_field_view.h"

#include <cstddef>
#include <span>
#include <vector>

/** Staged field updates with all-or-nothing publication.
 *
 * The invariant: no caller-visible value changes until every staged value has
 * passed local validation, and in the collective form until every rank has
 * agreed. Staging writes into transaction-private memory, so a rejected or
 * abandoned update cannot leave a field half-written.
 *
 * Accumulation is canonicalized rather than applied in place. Floating-point
 * addition is not associative, so summing contributions as they arrive makes
 * the published value depend on arrival order, which in turn makes it depend
 * on rank counts and scheduling. Contributions are therefore collected and
 * summed at commit in a total order that does not depend on staging order.
 */
namespace KARMA
{
enum class UpdateMode
{
    /** Every value must be staged exactly once; a second value for one slot is
     * ambiguous and rejected rather than resolved by arrival order. */
    Replace,
    /** Staged slots replace the published value and unstaged slots keep it; a
     * second value for one slot is still ambiguous and rejected. This is the
     * shape of a sparse insertion into an existing field. */
    SparseReplace,
    /** Contributions add to the published value; slots may be left unstaged. */
    Accumulate
};

/** Publication target. Implementations must publish all values or none. */
class FieldStorage
{
  public:
    virtual ~FieldStorage();

    [[nodiscard]] virtual std::size_t value_count() const = 0;
    /** Snapshot the currently published values, for accumulation bases and for
     * restoring a previous commit. */
    virtual OutcomeReason read_into(std::span<Real> destination) const = 0;
    /** Publish all values atomically from the caller's perspective. */
    virtual OutcomeReason publish(std::span<const Real> values) = 0;
};

/** Publishes into a caller-owned dense array. */
class HostArrayStorage final : public FieldStorage
{
  public:
    explicit HostArrayStorage(std::span<Real> target) noexcept : _target(target) {}

    [[nodiscard]] std::size_t value_count() const override { return _target.size(); }
    OutcomeReason read_into(std::span<Real> destination) const override;
    OutcomeReason publish(std::span<const Real> values) override;

  private:
    std::span<Real> _target;
};

/** Host-array storage that refuses publication, to prove that a failed commit
 * leaves no partial mutation. `failing_value` selects the value at which the
 * refusal happens, so the refusal occurs mid-write rather than up front. */
class FailingHostArrayStorage final : public FieldStorage
{
  public:
    FailingHostArrayStorage(std::span<Real> target, std::size_t failing_value) noexcept
        : _target(target), _failing_value(failing_value)
    {
    }

    [[nodiscard]] std::size_t value_count() const override { return _target.size(); }
    OutcomeReason read_into(std::span<Real> destination) const override;
    OutcomeReason publish(std::span<const Real> values) override;

  private:
    std::span<Real> _target;
    std::size_t _failing_value = 0;
};

class FieldTransaction
{
  public:
    FieldTransaction(FieldStorage& storage, FieldViewLayout layout, UpdateMode mode);

    /** Dense-index staging: the hot path. Performs no key lookup and, once the
     * transaction has been used at least once, no allocation. */
    [[nodiscard]] OutcomeReason stage(std::size_t entry, unsigned component, Real value) noexcept;

    /** Setup-path staging through a stable key. Counted by the index map. */
    [[nodiscard]] OutcomeReason stage_by_key(const FieldIndexMap& map, const FieldEntryKey& key, unsigned component,
                                             Real value);

    /** Local, total, and side-effect free. */
    [[nodiscard]] OutcomeReason validate_local() const noexcept;

    /** Nonthrowing. Validates, then publishes; on any failure nothing is
     * published and no caller-visible value changes. */
    [[nodiscard]] LocalOutcome commit() noexcept;

    /** Discards staged state. Infallible by construction: it touches only
     * transaction-private memory, which is why it is separate from
     * `restore_committed()`. */
    void rollback() noexcept;

    /** Republishes the snapshot taken before the last successful commit. This
     * can fail - it is a publication - which is exactly why discarding staged
     * state is a different operation. */
    [[nodiscard]] LocalOutcome restore_committed() noexcept;

    /** Validate on every rank, reconcile, and publish only if every rank
     * agreed. No rank publishes when any rank's validation failed.
     *
     * Every rank of `communicator` must call this. A publication failure after
     * a successful reconciliation is reported with the failing rank; the
     * distributed state is then inconsistent and the caller must restore, which
     * is the one case this API cannot make atomic.
     */
    [[nodiscard]] ReconciledOutcome commit_collective(MPI_Comm communicator) noexcept;

    [[nodiscard]] std::size_t staged_count() const noexcept;
    [[nodiscard]] const FieldViewLayout& layout() const noexcept { return _layout; }
    [[nodiscard]] UpdateMode mode() const noexcept { return _mode; }

  private:
    [[nodiscard]] OutcomeReason build_published_values(std::span<Real> destination) const noexcept;

    FieldStorage* _storage = nullptr;
    FieldViewLayout _layout;
    UpdateMode _mode = UpdateMode::Replace;

    /** Replacement staging: one value per slot plus a staged flag. */
    std::vector<Real> _staged;
    std::vector<bool> _has_staged;
    /** Accumulation staging: contributions kept until commit canonicalizes them. */
    std::vector<std::pair<std::size_t, Real>> _contributions;
    /** Snapshot of the values published before the last successful commit. */
    std::vector<Real> _previous;
    bool _has_previous = false;
    /** Scratch for the values a commit will publish, kept to avoid allocating
     * inside a warmed commit. */
    mutable std::vector<Real> _publication;
};
} // namespace KARMA

#endif
