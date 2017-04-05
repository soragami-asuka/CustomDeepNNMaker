//=======================================
// ���C���[DLL�֐�
//=======================================
#ifndef __GRAVISBELL_NN_LAYER_FUNCTION_H__
#define __GRAVISBELL_NN_LAYER_FUNCTION_H__

#include"../../Common/Guiddef.h"
#include"../../Common/VersionCode.h"
#include"../../Common/ErrorCode.h"

#include"ILayerDLLManager.h"
#include"INNLayer.h"


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
	typedef SettingData::Standard::IData* (*FuncCreateLayerStructureSettingFromBuffer)(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


	/** �w�K�ݒ���쐬���� */
	typedef SettingData::Standard::IData* (*FuncCreateLayerLearningSetting)(void);
	/** �w�K�ݒ���쐬����
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
		@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
	typedef SettingData::Standard::IData* (*FuncCreateLayerLearningSettingFromBuffer)(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


	/** CPU�����p�̃��C���[���쐬 */
	typedef INNLayer* (*FuncCreateLayerCPU)(Gravisbell::GUID guid, const ILayerDLLManager* pLayerDLLManager);

	/** GPU�����p�̃��C���[���쐬 */
	typedef INNLayer* (*FuncCreateLayerGPU)(Gravisbell::GUID guid, const ILayerDLLManager* pLayerDLLManager);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif