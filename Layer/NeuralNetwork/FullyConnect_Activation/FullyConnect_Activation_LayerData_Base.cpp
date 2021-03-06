//======================================
// 全結合ニューラルネットワークのレイヤーデータ
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_LayerData_Base.h"
#include"FullyConnect_Activation_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	FullyConnect_Activation_LayerData_Base::FullyConnect_Activation_LayerData_Base(const Gravisbell::GUID& guid)
		:	INNLayerData()
		,	guid	(guid)
		,	inputDataStruct	()	/**< 入力データ構造 */
		,	pLayerStructure	(NULL)	/**< レイヤー構造を定義したコンフィグクラス */
		,	layerStructure	()		/**< レイヤー構造 */
	{
	}
	/** デストラクタ */
	FullyConnect_Activation_LayerData_Base::~FullyConnect_Activation_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// 共通処理
	//===========================
	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID FullyConnect_Activation_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID FullyConnect_Activation_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** 設定情報を設定 */
	ErrorCode FullyConnect_Activation_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
	{
		Gravisbell::ErrorCode err = ERROR_CODE_NONE;

		// レイヤーコードを確認
		{
			Gravisbell::GUID config_guid;
			err = config.GetLayerCode(config_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			Gravisbell::GUID layer_guid;
			err = ::GetLayerCode(layer_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			if(config_guid != layer_guid)
				return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
		}

		if(this->pLayerStructure != NULL)
			delete this->pLayerStructure;
		this->pLayerStructure = config.Clone();

		// 構造体に読み込む
		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}

	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* FullyConnect_Activation_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 FullyConnect_Activation_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// 設定情報
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// 本体のバイト数
		bufferSize += (this->GetNeuronCount() * this->GetInputBufferCount()) * sizeof(NEURON_TYPE);	// ニューロン係数
		bufferSize += this->GetNeuronCount() * sizeof(NEURON_TYPE);	// バイアス係数


		return bufferSize;
	}


	//===========================
	// 入力レイヤー関連
	//===========================
	/** 入力データ構造を取得する.
		@return	入力データ構造 */
	IODataStruct FullyConnect_Activation_LayerData_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** 入力バッファ数を取得する. */
	U32 FullyConnect_Activation_LayerData_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
	}


	//===========================
	// 出力レイヤー関連
	//===========================
	/** 出力データ構造を取得する */
	IODataStruct FullyConnect_Activation_LayerData_Base::GetOutputDataStruct()const
	{
		IODataStruct outputDataStruct;

		outputDataStruct.x = 1;
		outputDataStruct.y = 1;
		outputDataStruct.z = 1;
		outputDataStruct.ch = this->GetNeuronCount();

		return outputDataStruct;
	}

	/** 出力バッファ数を取得する */
	unsigned int FullyConnect_Activation_LayerData_Base::GetOutputBufferCount()const
	{
		IODataStruct outputDataStruct = GetOutputDataStruct();

		return outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.ch;
	}

	
	//===========================
	// 固有関数
	//===========================
	/** ニューロン数を取得する */
	U32 FullyConnect_Activation_LayerData_Base::GetNeuronCount()const
	{
		return this->layerStructure.NeuronCount;
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
