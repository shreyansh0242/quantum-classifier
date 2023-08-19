import React, { useState } from 'react';
import { Box, Button, TextField, IconButton, Typography, useTheme } from "@mui/material";

import { tokens } from "../../theme";
import { Formik } from "formik";
import * as yup from "yup";
import useMediaQuery from "@mui/material/useMediaQuery";
import Header from "../../components/Header";
import FromBox from '../../components/FormBox';
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";

const ImageUploader = () => {
  const [leftEyeImage, setLeftEyeImage] = useState(null);
  const [rightEyeImage, setRightEyeImage] = useState(null);

  const handleLeftEyeUpload = (e) => {
    const file = e.target.files[0];
    setLeftEyeImage(file);
  };

  const handleRightEyeUpload = (e) => {
    const file = e.target.files[0];
    setRightEyeImage(file);
  };

  const handleSubmit = () => {
    // TODO: Send images to the backend
    // You'll need to use APIs like FormData to send the images to the server.
  };
  const theme = useTheme();
    const colors = tokens(theme.palette.mode);
  return (
    <Box
    
    >
        <Header title="Test Page" subtitle="Upload Images to get report" />
        <h2>Upload Your Eye Images</h2>
        {/* left eye box */}
        <Box
            gridColumn="span 3"
            backgroundColor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
        >
            <FromBox
                title="LEFT EYE"
                subtitle="Uplad High quality images for accurate results"
        
            />
            
            <Box>
                <label>Left Eye Image:</label>
                <input type="file" accept="image/*" onChange={handleLeftEyeUpload} />
            </Box>
            
        </Box>
        <br />
        {/* right eye box */}
        <Box
            gridColumn="span 3"
            backgroundColor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
        >
            <FromBox
                title="RIGHT EYE"
                subtitle="Uplad High quality images for accurate results"
        
            />
            <Box>
                <label>Right Eye Image:</label>
                <input type="file" accept="image/*" onChange={handleRightEyeUpload} />
            </Box>
            
        </Box>

        <Box 
            display="flex"
            alignItems="center"
            justifyContent="center"
        >
                <Button
                sx={{
                    backgroundColor: colors.blueAccent[700],
                    color: colors.grey[100],
                    fontSize: "14px",
                    fontWeight: "bold",
                    padding: "10px 20px",
                }}
                >
                <DownloadOutlinedIcon sx={{ mr: "10px" }} />
                Submit
                </Button>
            </Box>
          
                {/*Result box  */}
        <Box
            gridColumn="span 12"
            backgroundColor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
            // width="90%"
            // height = "100%"
        >   
            <Typography
                variant="h5"
                color={colors.greenAccent[500]}
                sx={{ mt: "15px" }}
              >
                Result section
              </Typography>
            
        </Box>
    </Box>
  );
};

export default ImageUploader;
