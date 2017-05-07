//=======================================
// エラーコード
//=======================================
#ifndef __GRAVISBELL_ERROR_CODE_H__
#define __GRAVISBELL_ERROR_CODE_H__

#include"Common.h"

namespace Gravisbell {

	enum ErrorCode : U32
	{
		ERROR_CODE_NONE = 0,

		// 共通系エラー
		ERROR_CODE_COMMON = 0x01000000,
		ERROR_CODE_COMMON_NULL_REFERENCE,			///< NULL参照
		ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE,		///< 配列外参照
		ERROR_CODE_COMMON_OUT_OF_VALUERANGE,		///< 値範囲の外
		ERROR_CODE_COMMON_ALLOCATION_MEMORY,		///< メモリの確保に失敗
		ERROR_CODE_COMMON_FILE_NOT_FOUND,			///< ファイルの参照に失敗
		ERROR_CODE_COMMON_CALCULATE_NAN,			///< 演算中にNANが発生した
		ERROR_CODE_COMMON_NOT_EXIST,				///< 存在しない
		ERROR_CODE_COMMON_ADD_ALREADY_SAMEID,		///< 既に同一IDが登録済み
		ERROR_CODE_COMMON_NOT_COMPATIBLE,			///< 未対応

		// DLL系エラー
		ERROR_CODE_DLL = 0x02000000,
		ERROR_CODE_DLL_LOAD_FUNCTION,			///< 関数の読み込みに失敗
		ERROR_CODE_DLL_ADD_ALREADY_SAMEID,		///< 既に同一IDのDLLが登録済み
		ERROR_CODE_DLL_ERASE_NOTFOUND,			///< 対象のID検索に失敗

		// 入出力系エラー
		ERROR_CODE_IO = 0x03000000,
		ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT,	///< 入出力のデータ数が一致しない

		// CUDA系エラー
		ERROR_CODE_CUDA = 0x04000000,
		ERROR_CODE_CUDA_INITIALIZE,				///< CUDAの初期化に失敗
		ERROR_CODE_CUDA_ALLOCATION_MEMORY,		///< メモリの確保に失敗
		ERROR_CODE_CUDA_COPY_MEMORY,			///< メモリのコピーに失敗
		ERROR_CODE_CUDA_CALCULATE,				///< 演算失敗


		// レイヤー系エラー
		ERROR_CODE_LAYER = 0x10000000,
		ERROR_CODE_NONREGIST_CONFIG,			///< 設定情報が登録されていない
		ERROR_CODE_FRAUD_INPUT_COUNT,			///< 入力数が不正
		ERROR_CODE_FRAUD_OUTPUT_COUNT,			///< 出力数が不正
		ERROR_CODE_FRAUD_NEURON_COUNT,			///< ニューロン数が不正
		// レイヤー追加系エラー
		ERROR_CODE_ADDLAYER = 0x10010000,
		ERROR_CODE_ADDLAYER_ALREADY_SAMEID,		///< 同じIDのレイヤーが既に登録済み
		ERROR_CODE_ADDLAYER_UPPER_LIMIT,		///< レイヤーの追加上限に達している
		ERROR_CODE_ADDLAYER_NOT_COMPATIBLE,		///< 未対応
		ERROR_CODE_ADDLAYER_NOT_EXIST,			///< レイヤーが存在しない
		// レイヤー削除系エラー
		ERROR_CODE_ERASELAYER = 0x10020000,
		ERROR_CODE_ERASELAYER_NOTFOUND,			///< 対象のID検索に失敗
		// レイヤー初期化系エラー
		ERROR_CODE_INITLAYER_DISAGREE_CONFIG,	///< 設定情報の型が不一致
		ERROR_CODE_INITLAYER_READ_CONFIG,		///< 設定情報の読み取りに失敗


		// ユーザー定義
		ERROR_CODE_USER = 0x80000000
	};

}	// Gravisbell

#endif