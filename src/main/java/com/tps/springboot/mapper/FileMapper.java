package com.tps.springboot.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.tps.springboot.entity.Files;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface FileMapper extends BaseMapper<Files> {
}
