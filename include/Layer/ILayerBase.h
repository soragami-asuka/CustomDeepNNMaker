//=======================================
// レイヤーベース
//=======================================
#ifndef __GRAVISBELL_I_LAYER_BASE_H__
#define __GRAVISBELL_I_LAYER_BASE_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"

#include"../SettingData/Standard/IData.h"

namespace Gravisbell {
namespace Layer {


	/** レイヤー種別 */
	enum ELayerKind : U32
	{
		LAYER_KIND_IOTYPE		 = 0xFF << 0,
		LAYER_KIND_INPUTTYPE	 = 0x01 << 0,
		LAYER_KIND_OUTPUTTYPE	 = 0x01 << 1,
		LAYER_KIND_SINGLE_INPUT  = (0x00 << 0) << 0,	/**< 入力レイヤー */
		LAYER_KIND_MULT_INPUT    = (0x01 << 0) << 0,	/**< 入力レイヤー */
		LAYER_KIND_SINGLE_OUTPUT = (0x00 << 1) << 0,	/**< 出力レイヤー */
		LAYER_KIND_MULT_OUTPUT   = (0x01 << 1) << 0,	/**< 出力レイヤー */

		LAYER_KIND_USETYPE		 = 0xFF << 8,
		LAYER_KIND_DATA			 = 0x01 << 8,	/**< データレイヤー.入出力 */
		LAYER_KIND_NEURALNETWORK = 0x02 << 8,	/**< ニューラルネットワークレイヤー */
		
		LAYER_KIND_CALCTYPE = 0x0F << 16,
		LAYER_KIND_UNKNOWN	= 0x00 << 16,
		LAYER_KIND_CPU		= 0x01 << 16,	/**< CPU処理レイヤー */
		LAYER_KIND_GPU		= 0x02 << 16,	/**< GPU処理レイヤー */

		LAYER_KIND_MEMORYTYPE		= 0x0F << 20,
		LAYER_KIND_UNKNOWNMEMORY	= 0x00 << 20,	
		LAYER_KIND_HOSTMEMORY		= 0x01 << 20,	/**< ホストメモリでデータのやりを取りを行う */
		LAYER_KIND_DEVICEMEMORY		= 0x02 << 20,	/**< デバイスメモリでデータのやりを取りを行う */
	};


	/** レイヤーベース */
	class ILayerBase
	{
	public:
		/** コンストラクタ */
		ILayerBase(){}
		/** デストラクタ */
		virtual ~ILayerBase(){}

	public:
		//=======================================
		// 共通処理
		//=======================================
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		virtual U32 GetLayerKind(void)const = 0;

		/** レイヤー固有のGUIDを取得する */
		virtual Gravisbell::GUID GetGUID(void)const = 0;

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		virtual Gravisbell::GUID GetLayerCode(void)const = 0;

		/** レイヤーの設定情報を取得する */
		virtual const SettingData::Standard::IData* GetLayerStructure()const = 0;

	public:
		//=======================================
		// 初期化処理
		//=======================================
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		virtual ErrorCode Initialize(void) = 0;


	public:
		//=======================================
		// 演算前処理
		//=======================================
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		virtual ErrorCode PreProcessLearn(unsigned int batchSize) = 0;
		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		virtual ErrorCode PreProcessCalculate(unsigned int batchSize) = 0;

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		virtual unsigned int GetBatchSize()const = 0;

	public:
		//=======================================
		// 演算ループ前処理
		//=======================================
		/** ループの初期化処理.データセットの実行開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		virtual ErrorCode PreProcessLoop() = 0;



		//====================================
		// 実行時設定
		//====================================
		/** 実行時設定を取得する. */
		virtual const SettingData::Standard::IData* GetRuntimeParameter()const = 0;
		virtual SettingData::Standard::IData* GetRuntimeParameter() = 0;

		/** 実行時設定を設定する.
			int型、float型、enum型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param) = 0;
		/** 実行時設定を設定する.
			int型、float型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param) = 0;
		/** 実行時設定を設定する.
			bool型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param) = 0;
		/** 実行時設定を設定する.
			string型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param) = 0;
	};

}	// Layer
}	// Gravisbell

#endif