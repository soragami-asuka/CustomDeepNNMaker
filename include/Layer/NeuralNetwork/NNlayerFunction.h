//=======================================
// ���C���[DLL�֐�
//=======================================
#ifndef __GRAVISBELL_NN_LAYER_FUNCTION_H__
#define __GRAVISBELL_NN_LAYER_FUNCTION_H__

#include"../../Common/Guiddef.h"
#include"../../Common/VersionCode.h"
#include"../../Common/ErrorCode.h"

#include"ILayerDLLManager.h"
#include"../ILayerData.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[���ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	typedef ErrorCode (*FuncGetLayerCode)(Gravisbell::GUID& o_layerCode);
	/** �o�[�W�����R�[�h���擾����.
		@param o_versionCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	typedef ErrorCode (*FuncGetVersionCode)(VersionCode& o_versionCode);


	/** ���C���[�\���ݒ���쐬���� */
	typedef SettingData::Standard::IData* (*FuncCreateLayerStructureSetting)(void);
	/** ���C���[�\���ݒ���쐬����
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
		@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
	typedef SettingData::Standard::IData* (*FuncCreateLayerStructureSettingFromBuffer)(const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize);


	/** �w�K�ݒ���쐬���� */
	typedef SettingData::Standard::IData* (*FuncCreateLayerRuntimeParameter)(void);
	/** �w�K�ݒ���쐬����
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
		@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
	typedef SettingData::Standard::IData* (*FuncCreateLayerRuntimeParameterFromBuffer)(const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize);


	/** ���C���[���쐬 */
	typedef ILayerData* (*FuncCreateLayerData)			(const ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const SettingData::Standard::IData& i_layerStructure);
	typedef ILayerData* (*FuncCreateLayerDataFromBuffer)(const ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif