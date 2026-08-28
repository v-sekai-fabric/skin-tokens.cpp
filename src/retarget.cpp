#include <skintokens/skintokens.hpp>

#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace skintokens {
namespace {

quat qmul(quat a, quat b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
            a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
            a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
            a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}

quat qnorm(quat value) {
    const float length=std::sqrt(value.x*value.x+value.y*value.y+value.z*value.z+value.w*value.w);
    return length>1.0e-12F ? quat{value.x/length,value.y/length,value.z/length,value.w/length} : quat{};
}

vec3 rotate(quat q, vec3 v) {
    q=qnorm(q);
    const vec3 u{q.x,q.y,q.z};
    const float dot=u.x*v.x+u.y*v.y+u.z*v.z;
    const vec3 cross{u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x};
    const float uu=u.x*u.x+u.y*u.y+u.z*u.z;
    return {2.0F*dot*u.x+(q.w*q.w-uu)*v.x+2.0F*q.w*cross.x,
            2.0F*dot*u.y+(q.w*q.w-uu)*v.y+2.0F*q.w*cross.y,
            2.0F*dot*u.z+(q.w*q.w-uu)*v.z+2.0F*q.w*cross.z};
}

vec3 add(vec3 a, vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
vec3 sub(vec3 a, vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
vec3 scale(vec3 a, float s) { return {a.x*s,a.y*s,a.z*s}; }
float length(vec3 a) { return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
vec3 normalized(vec3 a) { const float l=length(a); return l>1.0e-8F ? scale(a,1.0F/l) : vec3{1.0F,0.0F,0.0F}; }
float distance(vec3 a, vec3 b) { return length(sub(a,b)); }

result<void> validate(const skeleton & rig) {
    const std::size_t count=rig.names.size();
    if (count==0U || count>256U || rig.parents.size()!=count || rig.rest_positions.size()!=count)
        return std::unexpected(detail::fail(error_code::invalid_argument,"skeleton arrays must contain 1..256 matching joints"));
    std::size_t roots=0U;
    for (std::size_t i=0;i<count;++i) {
        if (rig.parents[i]<0) ++roots;
        else if (static_cast<std::size_t>(rig.parents[i])>=i)
            return std::unexpected(detail::fail(error_code::invalid_argument,"skeleton parents must precede children"));
    }
    if (roots!=1U) return std::unexpected(detail::fail(error_code::invalid_argument,"skeleton must contain exactly one root"));
    return {};
}

result<void> validate(const motion & value) {
    auto rig=validate(value.rig);
    if (!rig) return rig;
    if (value.frames==0U || value.root_translations.size()!=value.frames ||
        value.local_rotations.size()!=value.frames*value.rig.names.size())
        return std::unexpected(detail::fail(error_code::invalid_argument,"motion arrays are incomplete"));
    return {};
}

std::optional<std::size_t> find(const skeleton & rig, std::string_view name) {
    const auto value=std::find(rig.names.begin(),rig.names.end(),name);
    if (value==rig.names.end()) return std::nullopt;
    return static_cast<std::size_t>(value-rig.names.begin());
}

struct body_pair { std::string_view target; std::string_view source; };
constexpr std::array<body_pair,22> body_pairs{{
    {"mixamorig:Hips","Hips"},{"mixamorig:Spine","Spine1"},
    {"mixamorig:Spine1","Spine2"},{"mixamorig:Spine2","Chest"},
    {"mixamorig:Neck","Neck1"},{"mixamorig:Head","Head"},
    {"mixamorig:LeftShoulder","LeftShoulder"},{"mixamorig:LeftArm","LeftArm"},
    {"mixamorig:LeftForeArm","LeftForeArm"},{"mixamorig:LeftHand","LeftHand"},
    {"mixamorig:RightShoulder","RightShoulder"},{"mixamorig:RightArm","RightArm"},
    {"mixamorig:RightForeArm","RightForeArm"},{"mixamorig:RightHand","RightHand"},
    {"mixamorig:LeftUpLeg","LeftLeg"},{"mixamorig:LeftLeg","LeftShin"},
    {"mixamorig:LeftFoot","LeftFoot"},{"mixamorig:LeftToeBase","LeftToeBase"},
    {"mixamorig:RightUpLeg","RightLeg"},{"mixamorig:RightLeg","RightShin"},
    {"mixamorig:RightFoot","RightFoot"},{"mixamorig:RightToeBase","RightToeBase"},
}};

const std::array<std::string_view,30> finger_names{{
    "mixamorig:LeftHandThumb1","mixamorig:LeftHandThumb2","mixamorig:LeftHandThumb3",
    "mixamorig:LeftHandIndex1","mixamorig:LeftHandIndex2","mixamorig:LeftHandIndex3",
    "mixamorig:LeftHandMiddle1","mixamorig:LeftHandMiddle2","mixamorig:LeftHandMiddle3",
    "mixamorig:LeftHandRing1","mixamorig:LeftHandRing2","mixamorig:LeftHandRing3",
    "mixamorig:LeftHandPinky1","mixamorig:LeftHandPinky2","mixamorig:LeftHandPinky3",
    "mixamorig:RightHandIndex1","mixamorig:RightHandIndex2","mixamorig:RightHandIndex3",
    "mixamorig:RightHandThumb1","mixamorig:RightHandThumb2","mixamorig:RightHandThumb3",
    "mixamorig:RightHandMiddle1","mixamorig:RightHandMiddle2","mixamorig:RightHandMiddle3",
    "mixamorig:RightHandRing1","mixamorig:RightHandRing2","mixamorig:RightHandRing3",
    "mixamorig:RightHandPinky1","mixamorig:RightHandPinky2","mixamorig:RightHandPinky3",
}};

struct map_entry { std::size_t target; std::size_t source; };
result<std::vector<map_entry>> semantic_map(const skeleton & source,const skeleton & target) {
    std::vector<map_entry> output;
    output.reserve(body_pairs.size());
    for (const auto & pair:body_pairs) {
        const auto s=find(source,pair.source),t=find(target,pair.target);
        if (!s) return std::unexpected(detail::fail(error_code::invalid_argument,
            "SOMA30 skeleton is missing joint "+std::string{pair.source}));
        if (!t) return std::unexpected(detail::fail(error_code::invalid_argument,
            "Mixamo52 skeleton is missing joint "+std::string{pair.target}));
        output.push_back({*t,*s});
    }
    return output;
}

std::pair<vec3,float> center_scale(const skeleton & rig) {
    vec3 low=rig.rest_positions.front(),high=low;
    for (const auto p:rig.rest_positions) {
        low={std::min(low.x,p.x),std::min(low.y,p.y),std::min(low.z,p.z)};
        high={std::max(high.x,p.x),std::max(high.y,p.y),std::max(high.z,p.z)};
    }
    return {{(low.x+high.x)*.5F,(low.y+high.y)*.5F,(low.z+high.z)*.5F},
            std::max({high.x-low.x,high.y-low.y,high.z-low.z})};
}

std::vector<vec3> posed_joints(const motion & value,std::size_t frame) {
    const std::size_t count=value.rig.names.size();
    std::vector<vec3> positions(count);
    std::vector<quat> rotations(count);
    for (std::size_t joint=0;joint<count;++joint) {
        const auto parent=value.rig.parents[joint];
        const quat local=value.local_rotations[frame*count+joint];
        if (parent<0) {
            rotations[joint]=qnorm(local);
            positions[joint]=value.root_translations[frame];
        } else {
            const auto p=static_cast<std::size_t>(parent);
            rotations[joint]=qnorm(qmul(rotations[p],local));
            positions[joint]=add(positions[p],rotate(rotations[p],sub(value.rig.rest_positions[joint],value.rig.rest_positions[p])));
        }
    }
    return positions;
}

void add_errors(const motion & source,const motion & target,std::span<const map_entry> mapping,
                double & sum,float & maximum,std::size_t & samples) {
    const auto anatomical_scale=[](const skeleton & rig,std::string_view head_name,
                                   std::string_view left_toe_name,std::string_view right_toe_name) {
        const auto head=find(rig,head_name),left=find(rig,left_toe_name),right=find(rig,right_toe_name);
        if (!head || !left || !right) return center_scale(rig).second;
        const vec3 feet=scale(add(rig.rest_positions[*left],rig.rest_positions[*right]),.5F);
        return std::max(distance(rig.rest_positions[*head],feet),1.0e-8F);
    };
    const float source_scale=anatomical_scale(source.rig,"Head","LeftToeBase","RightToeBase");
    const float target_scale=anatomical_scale(target.rig,"mixamorig:Head","mixamorig:LeftToeBase","mixamorig:RightToeBase");
    for (std::size_t frame=0;frame<source.frames;++frame) {
        const auto source_pose=posed_joints(source,frame),target_pose=posed_joints(target,frame);
        const vec3 source_root=source_pose.front(),target_root=target_pose.front();
        for (const auto item:mapping) {
            const auto a=scale(sub(source_pose[item.source],source_root),1.0F/source_scale);
            const auto b=scale(sub(target_pose[item.target],target_root),1.0F/target_scale);
            const float error=distance(a,b);
            sum+=error;maximum=std::max(maximum,error);++samples;
        }
    }
}

} // namespace

result<skeleton> make_mixamo52_rig(const skeleton & source) {
    auto valid=validate(source);
    if (!valid) return std::unexpected(valid.error());
    skeleton output;
    output.names.reserve(52U);output.parents.reserve(52U);output.rest_positions.reserve(52U);
    const std::array<std::int32_t,22> parents{{-1,0,1,2,3,4,3,6,7,8,3,10,11,12,0,14,15,16,0,18,19,20}};
    for (std::size_t i=0;i<body_pairs.size();++i) {
        const auto source_joint=find(source,body_pairs[i].source);
        if (!source_joint) return std::unexpected(detail::fail(error_code::invalid_argument,
            "SOMA30 skeleton is missing joint "+std::string{body_pairs[i].source}));
        output.names.emplace_back(body_pairs[i].target);
        output.parents.push_back(parents[i]);
        output.rest_positions.push_back(source.rest_positions[*source_joint]);
    }
    const auto add_hand=[&](bool left,std::span<const std::string_view> names) -> result<void> {
        const std::string side=left?"Left":"Right";
        const auto hand=find(source,side+"Hand"),thumb=find(source,side+"HandThumbEnd"),middle=find(source,side+"HandMiddleEnd");
        if (!hand || !thumb || !middle) return std::unexpected(detail::fail(error_code::invalid_argument,
            "SOMA30 skeleton is missing "+side+" hand endpoints"));
        const vec3 origin=source.rest_positions[*hand];
        const vec3 middle_vector=sub(source.rest_positions[*middle],origin);
        const float hand_length=std::max(length(middle_vector),1.0e-5F);
        const vec3 forward=normalized(middle_vector);
        const vec3 thumb_vector=sub(source.rest_positions[*thumb],origin);
        vec3 lateral=sub(thumb_vector,scale(forward,thumb_vector.x*forward.x+thumb_vector.y*forward.y+thumb_vector.z*forward.z));
        if (length(lateral)<hand_length*.05F) lateral={left?0.0F:0.0F,0.0F,left?-1.0F:1.0F};
        lateral=normalized(lateral);
        const std::array<float,5> along{{.78F,.96F,1.0F,.92F,.82F}};
        const std::array<float,5> across{{.42F,.14F,0.0F,-.13F,-.27F}};
        const std::size_t hand_target=*find(output,left?"mixamorig:LeftHand":"mixamorig:RightHand");
        const std::array<std::size_t,5> semantic_order=left ?
            std::array<std::size_t,5>{{0U,1U,2U,3U,4U}} :
            std::array<std::size_t,5>{{1U,0U,2U,3U,4U}};
        for (std::size_t ordered=0;ordered<5U;++ordered) {
            const std::size_t finger=semantic_order[ordered];
            vec3 endpoint=add(origin,add(scale(forward,hand_length*along[finger]),scale(lateral,hand_length*across[finger])));
            if (finger==0U) endpoint=source.rest_positions[*thumb];
            if (finger==2U) endpoint=source.rest_positions[*middle];
            std::int32_t parent=static_cast<std::int32_t>(hand_target);
            for (std::size_t segment=0;segment<3U;++segment) {
                output.names.emplace_back(names[ordered*3U+segment]);
                output.parents.push_back(parent);
                output.rest_positions.push_back(add(origin,scale(sub(endpoint,origin),static_cast<float>(segment+1U)/3.0F)));
                parent=static_cast<std::int32_t>(output.names.size()-1U);
            }
        }
        return {};
    };
    auto left=add_hand(true,std::span<const std::string_view>{finger_names}.first<15>());
    if (!left) return std::unexpected(left.error());
    // The published Mixamo order puts right index before right thumb.
    auto right=add_hand(false,std::span<const std::string_view>{finger_names}.subspan(15U));
    if (!right) return std::unexpected(right.error());
    // Upstream's arbitrary-GLB path labels the asset "articulation", so its
    // Mixamo parts table is not selected. Blender's armature importer exposes
    // bones in hierarchy preorder instead. Skin-code group i is positional;
    // retaining the YAML's body-then-hands list here would decode hand codes
    // onto unrelated body joints even though the named retarget is correct.
    std::vector<std::size_t> order;
    order.reserve(output.names.size());
    const auto visit = [&](auto && self,std::size_t parent) -> void {
        order.push_back(parent);
        for (std::size_t child=0;child<output.parents.size();++child)
            if (output.parents[child]==static_cast<std::int32_t>(parent)) self(self,child);
    };
    visit(visit,0U);
    if (order.size()!=output.names.size())
        return std::unexpected(detail::fail(error_code::invalid_argument,"Mixamo52 hierarchy is disconnected"));
    std::vector<std::size_t> old_to_new(order.size());
    for (std::size_t index=0;index<order.size();++index) old_to_new[order[index]]=index;
    skeleton arranged;
    arranged.names.reserve(order.size());arranged.parents.reserve(order.size());arranged.rest_positions.reserve(order.size());
    for (const auto old:order) {
        arranged.names.push_back(output.names[old]);
        arranged.rest_positions.push_back(output.rest_positions[old]);
        arranged.parents.push_back(output.parents[old]<0 ? -1 :
            static_cast<std::int32_t>(old_to_new[static_cast<std::size_t>(output.parents[old])]));
    }
    return arranged;
}

result<motion> retarget_soma30_to_mixamo52(const motion & animation,const skeleton & target) {
    auto valid_source=validate(animation);if (!valid_source) return std::unexpected(valid_source.error());
    auto valid_target=validate(target);if (!valid_target) return std::unexpected(valid_target.error());
    auto mapping=semantic_map(animation.rig,target);if (!mapping) return std::unexpected(mapping.error());
    motion output;
    output.frames=animation.frames;output.frames_per_second=animation.frames_per_second;output.rig=target;
    output.root_translations=animation.root_translations;
    output.local_rotations.assign(output.frames*target.names.size(),{});
    const auto source_neck2=find(animation.rig,"Neck2");
    const auto target_neck=find(target,"mixamorig:Neck");
    for (std::size_t frame=0;frame<output.frames;++frame) {
        for (const auto item:*mapping)
            output.local_rotations[frame*target.names.size()+item.target]=
                animation.local_rotations[frame*animation.rig.names.size()+item.source];
        if (source_neck2 && target_neck) {
            auto & neck=output.local_rotations[frame*target.names.size()+*target_neck];
            neck=qnorm(qmul(neck,animation.local_rotations[frame*animation.rig.names.size()+*source_neck2]));
        }
    }
    return output;
}

result<retarget_report> validate_soma30_to_mixamo52(const motion & animation,const skeleton & target) {
    auto valid=validate(animation);if (!valid) return std::unexpected(valid.error());
    auto mapping=semantic_map(animation.rig,target);if (!mapping) return std::unexpected(mapping.error());
    auto transferred=retarget_soma30_to_mixamo52(animation,target);
    if (!transferred) return std::unexpected(transferred.error());
    retarget_report report;
    double sum=0.0;std::size_t samples=0U;
    add_errors(animation,*transferred,*mapping,sum,report.motion_max_position_error,samples);
    report.motion_mean_position_error=samples?static_cast<float>(sum/static_cast<double>(samples)):0.0F;

    constexpr float sine=.25881904510252074F,cosine=.9659258262890683F; // 30 degrees / 2
    const std::array<quat,6> probes{{{sine,0,0,cosine},{-sine,0,0,cosine},
        {0,sine,0,cosine},{0,-sine,0,cosine},{0,0,sine,cosine},{0,0,-sine,cosine}}};
    const auto source_neck2=find(animation.rig,"Neck2");
    double isolated_sum=0.0;std::size_t isolated_samples=0U;
    for (const auto item:*mapping) {
        // Neck is a deliberate two-to-one collapse and is exercised through
        // both source joints; all other probes are exact semantic pairs.
        std::array<std::size_t,2> drivers{{item.source,item.source}};
        std::size_t driver_count=1U;
        if (target.names[item.target]=="mixamorig:Neck" && source_neck2) { drivers[1]=*source_neck2;driver_count=2U; }
        for (std::size_t d=0;d<driver_count;++d) for (const auto probe:probes) {
            motion isolated;
            isolated.frames=1U;isolated.frames_per_second=30.0F;isolated.rig=animation.rig;
            isolated.root_translations={animation.rig.rest_positions.front()};
            isolated.local_rotations.assign(animation.rig.names.size(),{});
            isolated.local_rotations[drivers[d]]=probe;
            auto result=retarget_soma30_to_mixamo52(isolated,target);
            if (!result) return std::unexpected(result.error());
            add_errors(isolated,*result,*mapping,isolated_sum,report.isolated_max_position_error,isolated_samples);
            ++report.isolated_cases;
        }
    }
    report.isolated_mean_position_error=isolated_samples?
        static_cast<float>(isolated_sum/static_cast<double>(isolated_samples)):0.0F;

    const auto source_left=find(animation.rig,"LeftFoot"),source_right=find(animation.rig,"RightFoot");
    const auto target_left=find(target,"mixamorig:LeftFoot"),target_right=find(target,"mixamorig:RightFoot");
    double velocity_difference=0.0,velocity_reference=0.0;
    if (animation.frames>1U && source_left && source_right && target_left && target_right) {
        auto previous_source=posed_joints(animation,0U),previous_target=posed_joints(*transferred,0U);
        for (std::size_t frame=1U;frame<animation.frames;++frame) {
            auto current_source=posed_joints(animation,frame),current_target=posed_joints(*transferred,frame);
            for (const auto pair:std::array<std::pair<std::size_t,std::size_t>,2>{{{*source_left,*target_left},{*source_right,*target_right}}}) {
                const vec3 a=sub(current_source[pair.first],previous_source[pair.first]);
                const vec3 b=sub(current_target[pair.second],previous_target[pair.second]);
                velocity_difference+=distance(a,b);velocity_reference+=length(a);
            }
            previous_source=std::move(current_source);previous_target=std::move(current_target);
        }
    }
    report.foot_velocity_relative_error=static_cast<float>(velocity_difference/std::max(velocity_reference,1.0e-8));
    return report;
}

result<retarget_report> compare_soma30_to_mixamo52(const motion & source,const motion & target) {
    auto valid_source=validate(source);if (!valid_source) return std::unexpected(valid_source.error());
    auto valid_target=validate(target);if (!valid_target) return std::unexpected(valid_target.error());
    if (source.frames!=target.frames)
        return std::unexpected(detail::fail(error_code::invalid_argument,"retarget clips have different frame counts"));
    auto mapping=semantic_map(source.rig,target.rig);if (!mapping) return std::unexpected(mapping.error());
    retarget_report report;
    double sum=0.0;std::size_t samples=0U;
    add_errors(source,target,*mapping,sum,report.motion_max_position_error,samples);
    report.motion_mean_position_error=samples?static_cast<float>(sum/static_cast<double>(samples)):0.0F;
    const auto source_left=find(source.rig,"LeftFoot"),source_right=find(source.rig,"RightFoot");
    const auto target_left=find(target.rig,"mixamorig:LeftFoot"),target_right=find(target.rig,"mixamorig:RightFoot");
    double velocity_difference=0.0,velocity_reference=0.0;
    if (source.frames>1U && source_left && source_right && target_left && target_right) {
        auto previous_source=posed_joints(source,0U),previous_target=posed_joints(target,0U);
        for (std::size_t frame=1U;frame<source.frames;++frame) {
            auto current_source=posed_joints(source,frame),current_target=posed_joints(target,frame);
            for (const auto pair:std::array<std::pair<std::size_t,std::size_t>,2>{{{*source_left,*target_left},{*source_right,*target_right}}}) {
                const vec3 a=sub(current_source[pair.first],previous_source[pair.first]);
                const vec3 b=sub(current_target[pair.second],previous_target[pair.second]);
                velocity_difference+=distance(a,b);velocity_reference+=length(a);
            }
            previous_source=std::move(current_source);previous_target=std::move(current_target);
        }
    }
    report.foot_velocity_relative_error=static_cast<float>(velocity_difference/std::max(velocity_reference,1.0e-8));
    return report;
}

result<motion> retarget_motion_to_rig(const motion & animation,const skeleton & target) {
    auto valid_source=validate(animation);if (!valid_source) return std::unexpected(valid_source.error());
    auto valid_target=validate(target);if (!valid_target) return std::unexpected(valid_target.error());
    const auto source_bounds=center_scale(animation.rig),target_bounds=center_scale(target);
    if (source_bounds.second<=1.0e-8F || target_bounds.second<=1.0e-8F)
        return std::unexpected(detail::fail(error_code::invalid_argument,"cannot retarget a degenerate skeleton"));
    std::vector<vec3> source_positions,target_positions;
    for (const auto value:animation.rig.rest_positions) source_positions.push_back(scale(sub(value,source_bounds.first),1.0F/source_bounds.second));
    for (const auto value:target.rest_positions) target_positions.push_back(scale(sub(value,target_bounds.first),1.0F/target_bounds.second));
    const auto descendant=[&](std::size_t candidate,std::int32_t ancestor) {
        if (ancestor<0) return true;
        std::int32_t cursor=static_cast<std::int32_t>(candidate);
        while (cursor>=0) { if (cursor==ancestor) return true;cursor=animation.rig.parents[static_cast<std::size_t>(cursor)]; }
        return false;
    };
    std::vector<std::size_t> mapping(target.names.size(),0U);
    for (std::size_t joint=1U;joint<target.names.size();++joint) {
        const auto mapped_parent=mapping[static_cast<std::size_t>(target.parents[joint])];
        float best=std::numeric_limits<float>::infinity();
        for (std::size_t candidate=0;candidate<source_positions.size();++candidate) {
            if (!descendant(candidate,static_cast<std::int32_t>(mapped_parent))) continue;
            float score=distance(source_positions[candidate],target_positions[joint]);score*=score;
            if ((source_positions[candidate].x<-.03F)!=(target_positions[joint].x<-.03F) && std::abs(target_positions[joint].x)>.08F) score+=1.0F;
            if (score<best) { best=score;mapping[joint]=candidate; }
        }
    }
    motion output;
    output.frames=animation.frames;output.frames_per_second=animation.frames_per_second;output.rig=target;
    output.local_rotations.resize(output.frames*target.names.size());
    for (std::size_t frame=0;frame<output.frames;++frame) for (std::size_t joint=0;joint<target.names.size();++joint)
        output.local_rotations[frame*target.names.size()+joint]=animation.local_rotations[frame*animation.rig.names.size()+mapping[joint]];
    output.root_translations.resize(output.frames);
    const auto first=animation.root_translations.front(),root=target.rest_positions.front();
    const float travel=target_bounds.second/source_bounds.second;
    for (std::size_t frame=0;frame<output.frames;++frame) {
        const auto value=animation.root_translations[frame];
        output.root_translations[frame]={root.x+(value.x-first.x)*travel,root.y+(value.y-first.y)*travel,root.z+(value.z-first.z)*travel};
    }
    return output;
}

} // namespace skintokens
