//=======================================
// ���C���[DLL�N���X
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DLL_H__
#define __GRAVISBELL_I_NN_LAYER_DLL_H__

#include<guiddef.h>

#include"../../Common/VersionCode.h"

#include"INNLayer.h"
#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ILayerDLL
	{
	public:
		/** �R���X�g���N�^ */
		ILayerDLL(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerDLL(){}

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)const = 0;
		/** �o�[�W�����R�[�h���擾����.
			@param o_versionCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ErrorCode GetVersionCode(VersionCode& o_versionCode)const = 0;

		//==============================
		// ���C���[�\���쐬
		//==============================
		/** ���C���[�\���ݒ���쐬���� */
		virtual SettingData::Standard::IData* CreateLayerStructureSetting(void)const = 0;
		/** ���C���[�\���ݒ���쐬����
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;


		//==============================
		// �w�K�ݒ�쐬
		//==============================
		/** ���C���[�w�K�ݒ���쐬���� */
		virtual SettingData::Standard::IData* CreateLearningSetting(void)const = 0;
		/** ���C���[�w�K�ݒ���쐬����
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;


		//==============================
		// ���C���[�쐬
		//==============================
		/** ���C���[�f�[�^���쐬.
			GUID�͎������蓖��.
			@param	i_layerStructure	���C���[�\��.
			@param	i_inputDataStruct	���̓f�[�^�\��. */
		virtual ILayerData* CreateLayerData(const SettingData::Standard::IData& i_layerStructure, const IODataStruct& i_inputDataStruct)const = 0;
		/** ���C���[�f�[�^���쐬
			@param guid	�쐬���C���[��GUID
			@param	i_layerStructure	���C���[�\��.
			@param	i_inputDataStruct	���̓f�[�^�\��. */
		virtual ILayerData* CreateLayerData(const Gravisbell::GUID& guid, const SettingData::Standard::IData& i_layerStructure, const IODataStruct& i_inputDataStruct)const = 0;
		
		/** ���C���[���쐬.
			GUID�͎������蓖��.
			@param	i_lpBuffer		�ǂݎ��p�o�b�t�@.
			@param	i_bufferSize	�g�p�\�ȃo�b�t�@�T�C�Y.
			@param	o_useBufferSize	���ۂɎg�p�����o�b�t�@�T�C�Y. */
		virtual ILayerData* CreateLayerDataFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;
		/** ���C���[���쐬
			@param guid	�쐬���C���[��GUID */
		virtual ILayerData* CreateLayerDataFromBuffer(const Gravisbell::GUID& guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif